import { describe, expect, it, vi } from "vitest";
import {
  ACP_METHODS,
  SESSION_UPDATES,
  STOP_REASONS,
  type AcpStreamMessage,
} from "../../core/types.js";
import { createMockContext } from "../../test-utils/e2e-harness.js";
import {
  buildContinuationText,
  createPrematureTurnStopFix,
} from "./index.js";

function makeToolCallUpdate(
  sessionId: string,
  toolCallId: string,
  status: string,
  title?: string,
): AcpStreamMessage {
  return {
    jsonrpc: "2.0",
    method: ACP_METHODS.SESSION_UPDATE,
    params: {
      sessionId,
      update: {
        sessionUpdate: SESSION_UPDATES.TOOL_CALL_UPDATE,
        toolCallId,
        status,
        title,
      },
    },
  };
}

function makeToolCall(sessionId: string, toolCallId: string, title: string): AcpStreamMessage {
  return {
    jsonrpc: "2.0",
    method: ACP_METHODS.SESSION_UPDATE,
    params: {
      sessionId,
      update: {
        sessionUpdate: SESSION_UPDATES.TOOL_CALL,
        toolCallId,
        title,
        status: "in_progress",
      },
    },
  };
}

function makeMessageChunk(sessionId: string, text: string): AcpStreamMessage {
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

function makePromptResult(id: string | number, stopReason = "endTurn"): AcpStreamMessage {
  return {
    jsonrpc: "2.0",
    id,
    result: {
      stopReason,
    },
  };
}

describe("premature-turn-stop reproduction and verification", () => {
  const sessionId = "session-test-repro";

  it("problem: raw upstream terminates prompt turn after tool calls without emitting assistant text", () => {
    const inboundMessages: AcpStreamMessage[] = [];

    // 1. Tool execution completes
    const toolUpdate = makeToolCallUpdate(
      sessionId,
      "call_1",
      "completed",
      "Run pre-commit checks",
    );
    inboundMessages.push(toolUpdate);

    // 2. Upstream immediately completes prompt turn with empty AgentStep (stopReason=16)
    const promptResult = makePromptResult(1, "endTurn");
    inboundMessages.push(promptResult);

    // Check what was delivered to the client:
    const messageChunks = inboundMessages.filter(
      (m) =>
        "params" in m &&
        (m.params as { update?: { sessionUpdate?: string } })?.update?.sessionUpdate ===
          SESSION_UPDATES.AGENT_MESSAGE_CHUNK,
    );

    // DEMONSTRATION OF DEFECT:
    // Zero assistant messages emitted. Client UI enters idle state with complete silence.
    expect(messageChunks).toHaveLength(0);
    expect(inboundMessages.at(-1)).toEqual(promptResult);
  });

  it("solution: fix provides explicit system prompt steering to prevent silent turn completion", () => {
    const fix = createPrematureTurnStopFix();
    const instructions = fix.getSystemInstructions?.();

    expect(instructions).toBeDefined();
    expect(instructions?.length).toBeGreaterThan(0);
    expect(
      instructions?.some((i) =>
        i.includes("Never end a prompt turn without emitting a clear assistant text message"),
      ),
    ).toBe(true);
  });

  it("solution: buildContinuationText produces actionable continuation directives", () => {
    const text = buildContinuationText(15);
    expect(text).toContain("[Automated Continuation]");
    expect(text).toContain("15 tool calls");
    expect(text).toContain("summarize what you have accomplished");
  });

  it("solution: intercepts premature empty turn stop, dispatches continuation prompt upstream, and preserves turn", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];

    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    // 1. Client initiates prompt turn
    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 101,
        method: ACP_METHODS.SESSION_PROMPT,
        params: { sessionId, prompt: [{ type: "text", text: "Deploy cluster" }] },
      } as unknown as AcpStreamMessage,
      context,
    );

    // 2. Model executes tool calls
    await fix.onInbound?.(makeToolCall(sessionId, "t1", "Running k3s-entrypoint"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "t1", "completed"), context);

    // 3. Upstream tries to terminate turn prematurely with stopReason: "endTurn" and 0 text
    const emptyResult = makePromptResult(101, STOP_REASONS.END_TURN);
    const intercepted = await fix.onInbound?.(emptyResult, context);

    // Result suppressed from client!
    expect(intercepted).toEqual([]);

    // Continuation prompt dispatched upstream to child process
    expect(writtenToChild).toHaveLength(1);
    expect(writtenToChild[0]).toMatchObject({
      jsonrpc: "2.0",
      method: ACP_METHODS.SESSION_PROMPT,
      params: {
        sessionId,
        prompt: [{ type: "text" }],
      },
    });

    const promptText = (
      writtenToChild[0] as unknown as { params: { prompt: Array<{ text: string }> } }
    ).params.prompt[0]?.text;
    expect(promptText).toContain("[Automated Continuation]");
    expect(promptText).toContain("1 tool calls");

    // 4. Upstream responds to continuation with assistant text
    await fix.onInbound?.(
      makeMessageChunk(sessionId, "Cluster deploy finished successfully."),
      context,
    );

    // 5. Upstream completes continuation prompt
    const continuationPromptId = (writtenToChild[0] as unknown as { id: string }).id;
    const continuationComplete = makePromptResult(continuationPromptId, STOP_REASONS.END_TURN);
    const finalResult = await fix.onInbound?.(continuationComplete, context);

    // Client receives final result with original prompt ID restored!
    expect(finalResult).toHaveLength(1);
    expect(finalResult?.[0]).toMatchObject({
      jsonrpc: "2.0",
      id: 101,
      result: { stopReason: STOP_REASONS.END_TURN },
    });
  });

  it("solution: does not trigger continuation when turn already emitted assistant text", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 202,
        method: ACP_METHODS.SESSION_PROMPT,
        params: { sessionId, prompt: [{ type: "text", text: "Check status" }] },
      } as unknown as AcpStreamMessage,
      context,
    );

    // Tool call followed by text chunk
    await fix.onInbound?.(makeToolCall(sessionId, "t2", "Running status"), context);
    await fix.onInbound?.(makeMessageChunk(sessionId, "All services healthy."), context);

    // Upstream finishes
    const normalResult = makePromptResult(202, STOP_REASONS.END_TURN);
    const result = await fix.onInbound?.(normalResult, context);

    // Passthrough unchanged, no continuation sent
    expect(result).toEqual([normalResult]);
    expect(writtenToChild).toHaveLength(0);
  });

  it("solution: does not trigger continuation when cancelled by user", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 303,
        method: ACP_METHODS.SESSION_PROMPT,
        params: { sessionId, prompt: [{ type: "text", text: "Long task" }] },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "t3", "Running long job"), context);

    // User cancels
    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        method: ACP_METHODS.SESSION_CANCEL,
        params: { sessionId },
      } as unknown as AcpStreamMessage,
      context,
    );

    const cancelResult = makePromptResult(303, STOP_REASONS.CANCELLED);
    const result = await fix.onInbound?.(cancelResult, context);

    expect(result).toEqual([cancelResult]);
    expect(writtenToChild).toHaveLength(0);
  });

  it("solution: intercepts premature stop when model only emits whitespace chunks after tools", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 404,
        method: ACP_METHODS.SESSION_PROMPT,
        params: { sessionId, prompt: [{ type: "text", text: "Why do we need this?" }] },
      } as unknown as AcpStreamMessage,
      context,
    );

    // 10 tool calls executed
    for (let i = 0; i < 10; i++) {
      await fix.onInbound?.(makeToolCall(sessionId, `call_${i}`, `Tool ${i}`), context);
      await fix.onInbound?.(makeToolCallUpdate(sessionId, `call_${i}`, "completed"), context);
    }

    // Model only emits whitespace chunk ("\n")
    await fix.onInbound?.(makeMessageChunk(sessionId, "\n"), context);

    // Upstream attempts to finish turn
    const emptyResult = makePromptResult(404, "endTurn");
    const intercepted = await fix.onInbound?.(emptyResult, context);

    expect(intercepted).toEqual([]);
    expect(writtenToChild).toHaveLength(1);

    const promptText = (
      writtenToChild[0] as unknown as { params: { prompt: Array<{ text: string }> } }
    ).params.prompt[0]?.text;
    expect(promptText).toContain("[Automated Continuation]");
    expect(promptText).toContain("10 tool calls");
  });

  it("solution: does not trigger continuation when turn has no tool calls", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 505,
        method: ACP_METHODS.SESSION_PROMPT,
        params: { sessionId, prompt: [{ type: "text", text: "Pure question" }] },
      } as unknown as AcpStreamMessage,
      context,
    );

    // No tool calls, turn finishes
    const normalResult = makePromptResult(505, "end_turn");
    const result = await fix.onInbound?.(normalResult, context);

    expect(result).toEqual([normalResult]);
    expect(writtenToChild).toHaveLength(0);
  });

  it("solution: does not trigger continuation when session was cancelled via session/cancel", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 606,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Some prompt" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    // Tools ran
    await fix.onInbound?.(makeToolCall(sessionId, "call_stop_1", "Tool 1"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_stop_1", "completed"), context);

    // Client cancels session
    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        method: ACP_METHODS.SESSION_CANCEL,
        params: { sessionId },
      } as unknown as AcpStreamMessage,
      context,
    );

    // Turn completes with cancelled
    const stopResult = makePromptResult(606, STOP_REASONS.CANCELLED);
    const result = await fix.onInbound?.(stopResult, context);

    expect(result).toEqual([stopResult]);
    expect(writtenToChild).toHaveLength(0);
  });

  it("solution: correctly detects assistant text when emitted as an array of content blocks", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 707,
        method: ACP_METHODS.SESSION_PROMPT,
        params: { sessionId, prompt: [{ type: "text", text: "Check status" }] },
      } as unknown as AcpStreamMessage,
      context,
    );

    // Tools ran
    await fix.onInbound?.(makeToolCall(sessionId, "call_707", "Status check"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_707", "completed"), context);

    // Upstream emits content as an array of blocks (common in ACP)
    await fix.onInbound?.(
      {
        jsonrpc: "2.0",
        method: ACP_METHODS.SESSION_UPDATE,
        params: {
          sessionId,
          update: {
            sessionUpdate: SESSION_UPDATES.AGENT_MESSAGE_CHUNK,
            content: [{ type: "text", text: "# Status Report\nAll healthy." }],
          },
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    const normalResult = makePromptResult(707, "end_turn");
    const result = await fix.onInbound?.(normalResult, context);

    // Properly detected text, no continuation dispatched
    expect(result).toEqual([normalResult]);
    expect(writtenToChild).toHaveLength(0);
  });

  it("solution: prompt containing system instructions with 'wait' does not suppress continuation", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    // Prompt includes system instructions containing the word "wait" and "stop"
    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 808,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [
            {
              type: "text",
              text: "System instructions (replace earlier instructions). Please wait for settlement before dispatching.",
            },
            {
              type: "text",
              text: "<skills>If task matches skill, do not stop without loading.</skills>",
            },
            {
              type: "text",
              text: "Refactor database migrations",
            },
          ],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "call_808", "Migrate db"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_808", "completed"), context);

    // Empty stopReason end_turn
    const emptyResult = makePromptResult(808, STOP_REASONS.END_TURN);
    const intercepted = await fix.onInbound?.(emptyResult, context);

    // Should NOT be suppressed by the word "wait" in host instructions
    expect(intercepted).toEqual([]);
    expect(writtenToChild).toHaveLength(1);
    expect(
      (writtenToChild[0] as unknown as { params: { prompt: Array<{ text: string }> } }).params
        .prompt[0]?.text,
    ).toContain("[Automated Continuation]");
  });

  it("solution: triggers continuation when tools are emitted via tool_call_update only", async () => {
    const fix = createPrematureTurnStopFix();
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 909,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Run lint" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    // Tool call emitted ONLY as tool_call_update (no tool_call notification)
    await fix.onInbound?.(
      makeToolCallUpdate(sessionId, "call_909", "completed", "ESLint"),
      context,
    );

    const emptyResult = makePromptResult(909, STOP_REASONS.END_TURN);
    const intercepted = await fix.onInbound?.(emptyResult, context);

    expect(intercepted).toEqual([]);
    expect(writtenToChild).toHaveLength(1);
  });

  it("solution: synthesizes fallback status message when continuations are exhausted without assistant text", async () => {
    const fix = createPrematureTurnStopFix({ maxContinuations: 1 });
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 1010,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Run complex job" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "call_1010", "Job"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_1010", "completed"), context);

    // 1. Initial empty completion triggers continuation 1
    const emptyResult = makePromptResult(1010, STOP_REASONS.END_TURN);
    const intercepted = await fix.onInbound?.(emptyResult, context);
    expect(intercepted).toEqual([]);
    expect(writtenToChild).toHaveLength(1);

    // 2. Upstream finishes continuation 1 with 0 text again (maxContinuations=1 reached)
    const continuationPromptId = (writtenToChild[0] as unknown as { id: string }).id;
    const continuationResult = makePromptResult(continuationPromptId, STOP_REASONS.END_TURN);
    const finalResult = await fix.onInbound?.(continuationResult, context);

    // Should synthesize fallback agent_message_chunk so client does not fail with ACP_NO_VISIBLE_RESPONSE
    expect(finalResult).toHaveLength(2);
    expect(finalResult?.[0]).toMatchObject({
      jsonrpc: "2.0",
      method: ACP_METHODS.SESSION_UPDATE,
      params: {
        sessionId,
        update: {
          sessionUpdate: SESSION_UPDATES.AGENT_MESSAGE_CHUNK,
          content: {
            type: "text",
            text: expect.stringContaining("Completed 1 tool action(s)"),
          },
        },
      },
    });
    expect(finalResult?.[1]).toMatchObject({
      jsonrpc: "2.0",
      id: 1010,
      result: { stopReason: STOP_REASONS.END_TURN },
    });
  });

  it("solution: drops duplicate turn completion events while continuation prompt is in flight, avoiding concurrent prompt dispatch", async () => {
    const fix = createPrematureTurnStopFix({ maxContinuations: 2 });
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 2001,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Run background task" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "call_2001", "Grep"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_2001", "completed"), context);

    // 1. First empty endTurn arrives
    const emptyResult1 = makePromptResult(2001, STOP_REASONS.END_TURN);
    const intercepted1 = await fix.onInbound?.(emptyResult1, context);
    expect(intercepted1).toEqual([]);
    expect(writtenToChild).toHaveLength(1);

    // 2. Immediate duplicate empty endTurn arrives (e.g. within 2ms from settlement or race)
    const emptyResult2 = makePromptResult(2001, STOP_REASONS.END_TURN);
    const intercepted2 = await fix.onInbound?.(emptyResult2, context);
    // Must be dropped completely!
    expect(intercepted2).toEqual([]);
    // Crucially: no second continuation prompt dispatched!
    expect(writtenToChild).toHaveLength(1);

    // 3. Now continuation finishes with assistant text
    const contId = (writtenToChild[0] as unknown as { id: string }).id;
    await fix.onInbound?.(makeMessageChunk(sessionId, "Task summary here."), context);
    const finalResult = await fix.onInbound?.(
      makePromptResult(contId, STOP_REASONS.END_TURN),
      context,
    );

    expect(finalResult).toHaveLength(1);
    expect(finalResult?.[0]).toMatchObject({
      jsonrpc: "2.0",
      id: 2001,
      result: { stopReason: STOP_REASONS.END_TURN },
    });
  });

  it("solution: intercepts continuation RPC errors, synthesizes fallback message and returns end_turn mapped to original prompt id", async () => {
    const fix = createPrematureTurnStopFix({ maxContinuations: 2 });
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 3001,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Failing continuation turn" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "call_3001", "Grep"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_3001", "completed"), context);

    // Empty stop triggers continuation
    const emptyResult = makePromptResult(3001, STOP_REASONS.END_TURN);
    const intercepted = await fix.onInbound?.(emptyResult, context);
    expect(intercepted).toEqual([]);
    expect(writtenToChild).toHaveLength(1);

    const contId = (writtenToChild[0] as unknown as { id: string }).id;

    // Upstream returns JSON-RPC error for the continuation prompt
    const rpcError: AcpStreamMessage = {
      jsonrpc: "2.0",
      id: contId,
      error: {
        code: -32603,
        message: "A foreground turn is already active",
      },
    };

    const errorResult = await fix.onInbound?.(rpcError, context);
    // Should synthesize fallback message chunk and resolve original prompt cleanly
    expect(errorResult).toHaveLength(2);
    expect(errorResult?.[0]).toMatchObject({
      jsonrpc: "2.0",
      method: ACP_METHODS.SESSION_UPDATE,
      params: {
        sessionId,
        update: {
          sessionUpdate: SESSION_UPDATES.AGENT_MESSAGE_CHUNK,
          content: {
            type: "text",
            text: expect.stringContaining(
              "Completed 1 tool action(s), but continuation encountered an error",
            ),
          },
        },
      },
    });
    expect(errorResult?.[1]).toMatchObject({
      jsonrpc: "2.0",
      id: 3001,
      result: {
        stopReason: STOP_REASONS.END_TURN,
      },
    });
  });
  it("solution: multi-round continuation executes subsequent round when first continuation halts prematurely", async () => {
    const fix = createPrematureTurnStopFix({ maxContinuations: 2 });
    const context = createMockContext();
    const writtenToChild: AcpStreamMessage[] = [];
    context.writeToChild = vi.fn().mockImplementation(async (msg) => {
      writtenToChild.push(msg);
    });

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 4001,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Multi-round task" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "call_4001", "Step 1"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_4001", "completed"), context);

    // Initial empty stop triggers continuation 1
    const initialResult = makePromptResult(4001, STOP_REASONS.END_TURN);
    const intercepted1 = await fix.onInbound?.(initialResult, context);
    expect(intercepted1).toEqual([]);
    expect(writtenToChild).toHaveLength(1);

    const cont1Id = (writtenToChild[0] as unknown as { id: string }).id;

    // Continuation 1 executes another tool call, but also halts with empty text
    await fix.onInbound?.(makeToolCall(sessionId, "call_4002", "Step 2"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_4002", "completed"), context);

    const cont1Result = makePromptResult(cont1Id, STOP_REASONS.END_TURN);
    const intercepted2 = await fix.onInbound?.(cont1Result, context);

    // Must trigger continuation 2 (maxContinuations=2)
    expect(intercepted2).toEqual([]);
    expect(writtenToChild).toHaveLength(2);

    const cont2Id = (writtenToChild[1] as unknown as { id: string }).id;
    expect(cont2Id).not.toEqual(cont1Id);

    // Continuation 2 completes with assistant text
    await fix.onInbound?.(makeMessageChunk(sessionId, "All steps finished."), context);
    const finalResult = await fix.onInbound?.(
      makePromptResult(cont2Id, STOP_REASONS.END_TURN),
      context,
    );

    expect(finalResult).toHaveLength(1);
    expect(finalResult?.[0]).toMatchObject({
      jsonrpc: "2.0",
      id: 4001,
      result: { stopReason: STOP_REASONS.END_TURN },
    });
  });

  it("solution: synthesizes fallback status chunk when continuation dispatch throws", async () => {
    const fix = createPrematureTurnStopFix({ maxContinuations: 2 });
    const context = createMockContext();
    context.writeToChild = vi.fn().mockRejectedValue(new Error("Pipe closed"));

    await fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 5001,
        method: ACP_METHODS.SESSION_PROMPT,
        params: {
          sessionId,
          prompt: [{ type: "text", text: "Failing dispatch" }],
        },
      } as unknown as AcpStreamMessage,
      context,
    );

    await fix.onInbound?.(makeToolCall(sessionId, "call_5001", "Grep"), context);
    await fix.onInbound?.(makeToolCallUpdate(sessionId, "call_5001", "completed"), context);

    const emptyResult = makePromptResult(5001, STOP_REASONS.END_TURN);
    const result = await fix.onInbound?.(emptyResult, context);

    expect(result).toHaveLength(2);
    expect(result?.[0]).toMatchObject({
      jsonrpc: "2.0",
      method: ACP_METHODS.SESSION_UPDATE,
      params: {
        sessionId,
        update: {
          sessionUpdate: SESSION_UPDATES.AGENT_MESSAGE_CHUNK,
          content: {
            type: "text",
            text: expect.stringContaining("Completed 1 tool action(s)"),
          },
        },
      },
    });
    expect(result?.[1]).toMatchObject({
      jsonrpc: "2.0",
      id: 5001,
      result: { stopReason: STOP_REASONS.END_TURN },
    });
  });
});
