/**
 * Problem:
 * When upstream Antigravity executes tool calls and halts with an empty completion (stopReason=16,
 * no text and no tool call), upstream completes the prompt turn with a successful result without
 * emitting any `agent_message_chunk`. Editor clients (Paseo, Zed) transition the session to idle,
 * leaving the user with zero explanation and an apparent hang, forcing the user to manually ask
 * "You stopped, status report and next steps?".
 *
 * Solution:
 * In addition to baseline system prompt steering, intercepts empty `endTurn` completions following
 * tool execution at runtime. Automatically dispatches an automated continuation prompt upstream
 * directing the model to summarize its progress and continue executing the task to completion,
 * without synthesizing artificial assistant prose in the proxy.
 */

import {
  ACP_METHODS,
  SESSION_UPDATES,
  STOP_REASONS,
  isMethod,
  type AcpFix,
  type AcpStreamMessage,
  type InboundContext,
  type OutboundContext,
  type SessionUpdateParams,
} from "../../core/types.js";
import { extractSessionId, trackPendingRequestSession } from "../../core/session-cache.js";

export interface PrematureTurnStopOptions {
  maxContinuations?: number;
}

function extractBlockText(block: unknown): string {
  if (typeof block === "string") return block;
  if (block && typeof block === "object" && "text" in block && typeof block.text === "string") {
    return block.text;
  }
  return "";
}

export function extractContentText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map(extractBlockText).join("");
  }
  return extractBlockText(content);
}

interface SessionTurnState {
  activePromptId?: string | number | null | undefined;
  continuationPromptId?: string | number | undefined;
  toolCallCount: number;
  seenToolCallIds: Set<string>;
  assistantTextLength: number;
  continuationCount: number;
}

export class PrematureTurnStopTracker {
  private readonly sessions = new Map<string, SessionTurnState>();
  private readonly promptIdToSessionId = new Map<string | number, string>();

  get(sessionId: string): SessionTurnState | undefined {
    return this.sessions.get(sessionId);
  }

  private getOrCreate(sessionId: string): SessionTurnState {
    let state = this.sessions.get(sessionId);
    if (!state) {
      state = {
        toolCallCount: 0,
        seenToolCallIds: new Set(),
        assistantTextLength: 0,
        continuationCount: 0,
      };
      this.sessions.set(sessionId, state);
    }
    return state;
  }

  startTurn(sessionId: string, promptId?: string | number | null): void {
    const state = this.getOrCreate(sessionId);
    if (state.activePromptId !== undefined && state.activePromptId !== null) {
      this.promptIdToSessionId.delete(state.activePromptId);
    }
    if (state.continuationPromptId !== undefined) {
      this.promptIdToSessionId.delete(state.continuationPromptId);
    }
    state.activePromptId = promptId;
    state.continuationPromptId = undefined;
    state.toolCallCount = 0;
    state.seenToolCallIds.clear();
    state.assistantTextLength = 0;
    state.continuationCount = 0;
    if (promptId !== undefined && promptId !== null) {
      this.promptIdToSessionId.set(promptId, sessionId);
    }
  }

  getSessionIdForPrompt(promptId: string | number): string | undefined {
    return this.promptIdToSessionId.get(promptId);
  }

  recordToolCall(sessionId: string, toolCallId?: string): void {
    const state = this.getOrCreate(sessionId);
    if (toolCallId) {
      if (state.seenToolCallIds.has(toolCallId)) return;
      state.seenToolCallIds.add(toolCallId);
    }
    state.toolCallCount++;
  }

  recordAssistantText(sessionId: string, length: number): void {
    const state = this.getOrCreate(sessionId);
    state.assistantTextLength += length;
  }

  registerContinuationPrompt(sessionId: string, continuationPromptId: string | number): void {
    const s = this.sessions.get(sessionId);
    if (s) {
      s.continuationPromptId = continuationPromptId;
      this.promptIdToSessionId.set(continuationPromptId, sessionId);
    }
  }

  clearContinuationPrompt(sessionId: string): void {
    const s = this.sessions.get(sessionId);
    if (s && s.continuationPromptId !== undefined) {
      this.promptIdToSessionId.delete(s.continuationPromptId);
      s.continuationPromptId = undefined;
    }
  }

  clearSession(sessionId: string): void {
    const s = this.sessions.get(sessionId);
    if (s) {
      if (s.activePromptId !== undefined && s.activePromptId !== null) {
        this.promptIdToSessionId.delete(s.activePromptId);
      }
      if (s.continuationPromptId !== undefined) {
        this.promptIdToSessionId.delete(s.continuationPromptId);
      }
    }
    this.sessions.delete(sessionId);
  }

  clear(): void {
    this.sessions.clear();
    this.promptIdToSessionId.clear();
  }
}

export function buildContinuationText(toolCallCount: number): string {
  return (
    `[Automated Continuation]: You executed ${toolCallCount} tool calls but stopped without providing a status report or proceeding with the task. ` +
    `Please summarize what you have accomplished, evaluate the current state, and continue executing the next steps to complete the task.`
  );
}

function createFallbackChunk(sessionId: string, text: string): AcpStreamMessage {
  return {
    jsonrpc: "2.0",
    method: ACP_METHODS.SESSION_UPDATE,
    params: {
      sessionId,
      update: {
        sessionUpdate: SESSION_UPDATES.AGENT_MESSAGE_CHUNK,
        content: { type: "text", text },
      },
    },
  };
}

async function dispatchContinuationPrompt(
  sessionId: string,
  tracker: PrematureTurnStopTracker,
  context: InboundContext,
): Promise<boolean> {
  const state = tracker.get(sessionId);
  if (!state || state.activePromptId === undefined || state.activePromptId === null) return false;

  state.continuationCount++;
  const continuationText = buildContinuationText(state.toolCallCount);
  const continuationPromptId = `continue_${sessionId}_${Date.now()}_${state.continuationCount}`;
  tracker.registerContinuationPrompt(sessionId, continuationPromptId);
  if (context.sessionCache) {
    trackPendingRequestSession(context.sessionCache, continuationPromptId, sessionId);
  }

  try {
    await context.writeToChild({
      jsonrpc: "2.0",
      id: continuationPromptId,
      method: ACP_METHODS.SESSION_PROMPT,
      params: {
        sessionId,
        prompt: [{ type: "text", text: continuationText }],
      },
    } as unknown as AcpStreamMessage);
    return true;
  } catch (err) {
    tracker.clearContinuationPrompt(sessionId);
    console.error(`[refined-antigravity-acp] Failed to send automated continuation prompt:`, err);
    return false;
  }
}

function resolveSessionId(
  msg: AcpStreamMessage,
  tracker: PrematureTurnStopTracker,
): string | undefined {
  const fromMsg = extractSessionId(msg);
  if (fromMsg) return fromMsg;
  const id = (msg as { id?: string | number | null }).id;
  return id !== null && id !== undefined ? tracker.getSessionIdForPrompt(id) : undefined;
}

function handleInboundUpdate(
  msg: AcpStreamMessage,
  sessionId: string,
  tracker: PrematureTurnStopTracker,
): void {
  const update =
    "params" in msg ? (msg.params as SessionUpdateParams | undefined)?.update : undefined;
  if (!update) return;

  if (update.sessionUpdate === SESSION_UPDATES.TOOL_CALL) {
    const toolCallId =
      typeof (update as { toolCallId?: unknown }).toolCallId === "string"
        ? (update as { toolCallId: string }).toolCallId
        : undefined;
    tracker.recordToolCall(sessionId, toolCallId);
  } else if (update.sessionUpdate === SESSION_UPDATES.TOOL_CALL_UPDATE) {
    if (typeof (update as { toolCallId?: unknown }).toolCallId === "string") {
      tracker.recordToolCall(sessionId, (update as { toolCallId: string }).toolCallId);
    }
  } else if (update.sessionUpdate === SESSION_UPDATES.AGENT_MESSAGE_CHUNK) {
    const text = extractContentText(update.content);
    if (text.trim().length > 0) {
      tracker.recordAssistantText(sessionId, text.trim().length);
    }
  }
}

function extractErrorMessage(error: unknown): string {
  const msg = (error as { message?: unknown } | null)?.message;
  return typeof msg === "string" ? msg : "upstream error";
}

function handlePromptError(
  msg: AcpStreamMessage,
  sessionId: string,
  tracker: PrematureTurnStopTracker,
): AcpStreamMessage[] | null {
  if (!("error" in msg) || !msg.error) return null;
  const state = tracker.get(sessionId);
  if (!state?.continuationPromptId) return null;

  const promptId = (msg as { id?: string | number }).id;
  if (promptId !== state.continuationPromptId) return null;

  const originalPromptId = state.activePromptId;
  const toolCallCount = state.toolCallCount;
  tracker.clearSession(sessionId);

  const errMsg = extractErrorMessage(msg.error);
  const fallbackChunk = createFallbackChunk(
    sessionId,
    `[Completed ${toolCallCount} tool action(s), but continuation encountered an error: ${errMsg}.]`,
  );
  const syntheticResult: AcpStreamMessage = {
    jsonrpc: "2.0",
    id: originalPromptId ?? null,
    result: { stopReason: STOP_REASONS.END_TURN },
  };
  return [fallbackChunk, syntheticResult];
}

function synthesizeFallbackResult(
  sessionId: string,
  state: SessionTurnState,
  tracker: PrematureTurnStopTracker,
): AcpStreamMessage[] {
  const toolCallCount = state.toolCallCount;
  const originalPromptId = state.activePromptId;
  tracker.clearSession(sessionId);
  const fallbackChunk = createFallbackChunk(
    sessionId,
    `[Completed ${toolCallCount} tool action(s), but no concluding summary was provided by the agent.]`,
  );
  const syntheticResult: AcpStreamMessage = {
    jsonrpc: "2.0",
    id: originalPromptId ?? null,
    result: { stopReason: STOP_REASONS.END_TURN },
  };
  return [fallbackChunk, syntheticResult];
}

async function handlePrematureStop(
  sessionId: string,
  state: SessionTurnState,
  isContinuationResponse: boolean,
  tracker: PrematureTurnStopTracker,
  context: InboundContext,
  maxContinuations: number,
): Promise<AcpStreamMessage[]> {
  if (state.continuationCount < maxContinuations) {
    if (isContinuationResponse) {
      tracker.clearContinuationPrompt(sessionId);
    }
    const dispatched = await dispatchContinuationPrompt(sessionId, tracker, context);
    if (dispatched) return [];
  }
  return synthesizeFallbackResult(sessionId, state, tracker);
}

function classifyPrompt(
  state: SessionTurnState,
  promptId: string | number | null | undefined,
): "duplicate" | "continuation" | "original" | "ignore" {
  if (promptId === undefined || promptId === null) return "ignore";
  if (state.continuationPromptId !== undefined) {
    if (promptId === state.activePromptId) return "duplicate";
    if (promptId === state.continuationPromptId) return "continuation";
    return "ignore";
  }
  return promptId === state.activePromptId ? "original" : "ignore";
}

function isCancelledStop(stopReason: unknown): boolean {
  return stopReason === STOP_REASONS.CANCELLED || stopReason === "cancelled";
}

function isEndTurnStop(stopReason: unknown): boolean {
  return stopReason === STOP_REASONS.END_TURN || stopReason === "endTurn";
}

function shouldContinue(stopReason: unknown, state: SessionTurnState): boolean {
  if (!isEndTurnStop(stopReason)) return false;
  return state.toolCallCount > 0 && state.assistantTextLength === 0;
}

function resolveContinuationResult(
  msg: AcpStreamMessage,
  state: SessionTurnState,
  tracker: PrematureTurnStopTracker,
  sessionId: string,
): AcpStreamMessage[] {
  const originalPromptId = state.activePromptId;
  tracker.clearSession(sessionId);
  return [{ ...msg, id: originalPromptId ?? null }];
}

async function handlePromptResult(
  msg: AcpStreamMessage,
  sessionId: string,
  tracker: PrematureTurnStopTracker,
  context: InboundContext,
  options: PrematureTurnStopOptions,
): Promise<AcpStreamMessage[] | null> {
  if (!("result" in msg)) return null;

  const res = msg.result as Record<string, unknown> | null;
  const stopReason = res?.stopReason;
  if (isCancelledStop(stopReason)) {
    tracker.clearSession(sessionId);
    return null;
  }

  const state = tracker.get(sessionId);
  if (!state) return null;

  const promptId = (msg as { id?: string | number | null }).id;
  const kind = classifyPrompt(state, promptId);
  if (kind === "duplicate") return [];
  if (kind === "ignore") return null;

  const isContinuation = kind === "continuation";
  if (shouldContinue(stopReason, state)) {
    const maxContinuations = options.maxContinuations ?? 2;
    return handlePrematureStop(
      sessionId,
      state,
      isContinuation,
      tracker,
      context,
      maxContinuations,
    );
  }

  if (isContinuation) {
    return resolveContinuationResult(msg, state, tracker, sessionId);
  }

  tracker.clearSession(sessionId);
  return null;
}

export function createPrematureTurnStopFix(options: PrematureTurnStopOptions = {}): AcpFix {
  const tracker = new PrematureTurnStopTracker();

  return {
    name: "premature-turn-stop",
    description:
      "Steers model and intercepts premature empty turn stops without assistant messages to continue e2e execution",

    getSystemInstructions(): readonly string[] {
      return [
        "Never end a prompt turn without emitting a clear assistant text message summarizing completed actions, current status, or next steps.",
      ];
    },

    onOutbound(msg: AcpStreamMessage, _context: OutboundContext) {
      const sessionId = extractSessionId(msg);
      if (!sessionId) return msg;

      if (isMethod(msg, ACP_METHODS.SESSION_PROMPT)) {
        const promptId = (msg as { id?: string | number }).id;
        tracker.startTurn(sessionId, promptId);
      } else if (
        isMethod(msg, ACP_METHODS.SESSION_CANCEL) ||
        isMethod(msg, ACP_METHODS.SESSION_CLOSE) ||
        isMethod(msg, ACP_METHODS.SESSION_DELETE)
      ) {
        tracker.clearSession(sessionId);
      }

      return msg;
    },

    async onInbound(msg: AcpStreamMessage, context: InboundContext) {
      const sessionId = resolveSessionId(msg, tracker);
      if (!sessionId) return [msg];

      if (isMethod(msg, ACP_METHODS.SESSION_UPDATE)) {
        handleInboundUpdate(msg, sessionId, tracker);
      }

      const promptHandled = await handlePromptResult(msg, sessionId, tracker, context, options);
      if (promptHandled) return promptHandled;

      const errorHandled = handlePromptError(msg, sessionId, tracker);
      if (errorHandled) return errorHandled;

      return [msg];
    },
  };
}

export const prematureTurnStopFix = createPrematureTurnStopFix();
