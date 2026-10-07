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

export function extractUserPromptText(content: unknown): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    for (let i = content.length - 1; i >= 0; i--) {
      const text = extractBlockText(content[i]).trim();
      if (text) {
        return text;
      }
    }
  }
  return extractContentText(content).trim();
}

const STOP_INTENT_PATTERN = /^\s*(?:stop|halt|pause|hold\s+on)(?:[\s!.,:;-]|$)/i;

export function isStopIntent(text: string): boolean {
  const trimmed = text.trim();
  return trimmed.length < 50 && STOP_INTENT_PATTERN.test(trimmed);
}

interface SessionTurnState {
  activePromptId?: string | number | null | undefined;
  continuationPromptId?: string | number | undefined;
  userAskedToStop: boolean;
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
        userAskedToStop: false,
        toolCallCount: 0,
        seenToolCallIds: new Set(),
        assistantTextLength: 0,
        continuationCount: 0,
      };
      this.sessions.set(sessionId, state);
    }
    return state;
  }

  startTurn(sessionId: string, promptId?: string | number | null, userAskedToStop = false): void {
    const state = this.getOrCreate(sessionId);
    if (state.continuationPromptId !== undefined) {
      this.promptIdToSessionId.delete(state.continuationPromptId);
    }
    state.activePromptId = promptId;
    state.continuationPromptId = undefined;
    state.userAskedToStop = userAskedToStop;
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

  isPrematureStop(
    sessionId: string,
    promptId: string | number | null | undefined,
    maxContinuations: number,
  ): boolean {
    const state = this.sessions.get(sessionId);
    if (!state || state.continuationPromptId !== undefined) return false;
    if (state.activePromptId === undefined || state.activePromptId === null) return false;

    return (
      state.activePromptId === promptId &&
      !state.userAskedToStop &&
      state.toolCallCount > 0 &&
      state.assistantTextLength === 0 &&
      state.continuationCount < maxContinuations
    );
  }

  incrementContinuationCount(sessionId: string): void {
    const s = this.sessions.get(sessionId);
    if (s) s.continuationCount++;
  }

  isContinuationInFlight(sessionId: string): boolean {
    return this.sessions.get(sessionId)?.continuationPromptId !== undefined;
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

  getActivePromptId(sessionId: string): string | number | null | undefined {
    return this.sessions.get(sessionId)?.activePromptId;
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
  const originalPromptId = tracker.getActivePromptId(sessionId);
  if (originalPromptId === undefined || originalPromptId === null) return false;

  tracker.incrementContinuationCount(sessionId);
  const toolCallCount = tracker.get(sessionId)?.toolCallCount ?? 0;
  const continuationText = buildContinuationText(toolCallCount);

  const continuationPromptId = `continue_${sessionId}_${Date.now()}`;
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

  if (
    update.sessionUpdate === SESSION_UPDATES.TOOL_CALL ||
    update.sessionUpdate === SESSION_UPDATES.TOOL_CALL_UPDATE
  ) {
    const toolCallId =
      typeof (update as { toolCallId?: unknown }).toolCallId === "string"
        ? (update as { toolCallId: string }).toolCallId
        : undefined;
    tracker.recordToolCall(sessionId, toolCallId);
  } else if (update.sessionUpdate === SESSION_UPDATES.AGENT_MESSAGE_CHUNK) {
    const text = extractContentText(update.content);
    if (text.trim().length > 0) {
      tracker.recordAssistantText(sessionId, text.trim().length);
    }
  }
}

function handleContinuationResponse(
  msg: AcpStreamMessage,
  promptId: string | number | null | undefined,
  sessionId: string,
  tracker: PrematureTurnStopTracker,
): AcpStreamMessage[] | null {
  if (promptId === undefined || promptId === null) return null;
  const state = tracker.get(sessionId);
  if (state?.continuationPromptId !== undefined && state.continuationPromptId === promptId) {
    const originalPromptId = state.activePromptId;
    tracker.clearContinuationPrompt(sessionId);
    return [{ ...msg, id: originalPromptId ?? null }];
  }
  return null;
}

function extractErrorMessage(error: unknown): string {
  if (typeof error === "object" && error !== null && "message" in error) {
    const msg = (error as { message?: unknown }).message;
    return typeof msg === "string" ? msg : "upstream error";
  }
  return "upstream error";
}

function handlePromptError(
  msg: AcpStreamMessage,
  sessionId: string,
  tracker: PrematureTurnStopTracker,
): AcpStreamMessage[] | null {
  if (!("error" in msg) || !msg.error) return null;
  const promptId = (msg as { id?: string | number | null }).id;
  if (promptId === undefined || promptId === null) return null;

  const state = tracker.get(sessionId);
  if (
    !state ||
    state.continuationPromptId === undefined ||
    state.continuationPromptId !== promptId
  ) {
    return null;
  }

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
    result: {
      stopReason: STOP_REASONS.END_TURN,
    },
  };
  return [fallbackChunk, syntheticResult];
}

function isStaleDuplicateCompletion(
  tracker: PrematureTurnStopTracker,
  sessionId: string,
  promptId: string | number | null | undefined,
): boolean {
  return (
    tracker.isContinuationInFlight(sessionId) &&
    promptId !== undefined &&
    promptId !== null &&
    promptId === tracker.getActivePromptId(sessionId)
  );
}

function isCancelledStop(stopReason: unknown): boolean {
  return stopReason === STOP_REASONS.CANCELLED || stopReason === "cancelled";
}

function isEndTurnStop(stopReason: unknown): boolean {
  return stopReason === STOP_REASONS.END_TURN || stopReason === "endTurn";
}

async function maybeDispatchContinuation(
  sessionId: string,
  promptId: string | number | null | undefined,
  isEndTurn: boolean,
  tracker: PrematureTurnStopTracker,
  context: InboundContext,
  maxContinuations: number,
): Promise<boolean> {
  if (isEndTurn && tracker.isPrematureStop(sessionId, promptId, maxContinuations)) {
    return dispatchContinuationPrompt(sessionId, tracker, context);
  }
  return false;
}

function finalizeContinuationResponse(
  continuationResponse: AcpStreamMessage[],
  sessionId: string,
  isEndTurn: boolean,
  tracker: PrematureTurnStopTracker,
): AcpStreamMessage[] {
  const state = tracker.get(sessionId);
  const needsFallback =
    isEndTurn &&
    state &&
    state.assistantTextLength === 0 &&
    state.toolCallCount > 0 &&
    !state.userAskedToStop;

  if (needsFallback) {
    const toolCallCount = state.toolCallCount;
    tracker.clearSession(sessionId);
    const fallbackChunk = createFallbackChunk(
      sessionId,
      `[Completed ${toolCallCount} tool action(s), but no concluding summary was provided by the agent.]`,
    );
    return [fallbackChunk, ...continuationResponse];
  }

  tracker.clearSession(sessionId);
  return continuationResponse;
}

async function handlePromptResult(
  msg: AcpStreamMessage,
  sessionId: string,
  tracker: PrematureTurnStopTracker,
  context: InboundContext,
  options: PrematureTurnStopOptions,
): Promise<AcpStreamMessage[] | null> {
  if (!("result" in msg)) return null;

  const promptId = (msg as { id?: string | number | null }).id;
  const res = msg.result as Record<string, unknown> | null;
  const stopReason = res?.stopReason;

  if (isCancelledStop(stopReason)) {
    tracker.clearSession(sessionId);
    return null;
  }

  if (isStaleDuplicateCompletion(tracker, sessionId, promptId)) {
    return [];
  }

  const isEndTurn = isEndTurnStop(stopReason);
  const continuationDispatched = await maybeDispatchContinuation(
    sessionId,
    promptId,
    isEndTurn,
    tracker,
    context,
    options.maxContinuations ?? 2,
  );
  if (continuationDispatched) return [];

  const continuationResponse = handleContinuationResponse(msg, promptId, sessionId, tracker);
  if (continuationResponse) {
    return finalizeContinuationResponse(continuationResponse, sessionId, isEndTurn, tracker);
  }

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
        "When executing commit commands, linter checks, or tests that encounter errors, explicitly explain the failures to the user before attempting further autonomous repairs.",
      ];
    },

    onOutbound(msg: AcpStreamMessage, _context: OutboundContext) {
      const sessionId = extractSessionId(msg);
      if (!sessionId) return msg;

      if (isMethod(msg, ACP_METHODS.SESSION_PROMPT)) {
        const promptId = (msg as { id?: string | number }).id;
        const promptBlocks = (msg as { params?: { prompt?: unknown } })?.params?.prompt;
        const userText = extractUserPromptText(promptBlocks);
        const askedToStop = isStopIntent(userText);
        tracker.startTurn(sessionId, promptId, askedToStop);
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
