import { describe, expect, it } from "vitest";
import type {
  AcpStreamMessage,
  InboundContext,
  OutboundContext,
  StderrContext,
} from "../../core/types.js";
import {
  createBackgroundTasksFix,
  detectCompletedSubagentsInText,
  formatScheduleTask,
  formatSubagentContent,
  inferToolName,
  parseSubagentStatusReports,
  parseSubagentsFromArgs,
  reconstructPlanFromSteps,
  type PlanEntry,
  type TrackedSubagentInfo,
} from "./index.js";

const dummyInboundContext = {} as InboundContext;

describe("inferToolName", () => {
  it("identifies subagent execution when tool name is invoke_subagent or schedule", () => {
    expect(inferToolName("invoke_subagent", undefined, undefined)).toBe("invoke_subagent");
    expect(inferToolName("schedule", undefined, undefined)).toBe("schedule");
  });

  it("identifies subagent execution when tool title describes running subagents or schedule", () => {
    expect(inferToolName(undefined, "Running invoke_subagent", undefined)).toBe("invoke_subagent");
    expect(inferToolName(undefined, "Run invoke_subagent?", undefined)).toBe("invoke_subagent");
    expect(inferToolName(undefined, "Running schedule", undefined)).toBe("schedule");
  });

  it("identifies subagent execution when tool arguments contain subagent declarations", () => {
    expect(
      inferToolName(undefined, undefined, {
        Subagents: [{ Role: "Worker", TypeName: "research" }],
      }),
    ).toBe("invoke_subagent");
    expect(
      inferToolName(undefined, undefined, JSON.stringify({ Subagents: [{ Role: "Worker" }] })),
    ).toBe("invoke_subagent");
  });

  it("ignores regular non-subagent tool calls", () => {
    expect(inferToolName("run_command", "Run ls -la", { CommandLine: "ls" })).toBeNull();
    expect(inferToolName(undefined, "view_file", { path: "foo.ts" })).toBeNull();
  });
});

describe("parseSubagentsFromArgs and formatSubagentContent", () => {
  it("formats human-readable task descriptions for each declared subagent", () => {
    const raw = {
      Subagents: [
        { Role: "Inspector", Prompt: "Check things", TypeName: "research" },
        { TypeName: "reviewer", Prompt: "Review code" },
        { Prompt: "Do background analysis" },
      ],
    };
    const subs = parseSubagentsFromArgs(raw);
    expect(subs).toHaveLength(3);
    const [sub0, sub1, sub2] = subs;
    expect(sub0).toBeDefined();
    expect(sub1).toBeDefined();
    expect(sub2).toBeDefined();
    expect(formatSubagentContent(sub0!)).toBe("Subagent: Inspector");
    expect(formatSubagentContent(sub1!)).toBe("Subagent: reviewer");
    expect(formatSubagentContent(sub2!)).toBe("Subagent: Do background analysis");
  });
});

describe("backgroundTasksFix processInbound", () => {
  it("surfaces running subagents as active checklist items in the execution plan", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_subagents";

    const msg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub_1",
          title: "Running invoke_subagent",
          kind: "other",
          rawInput: {
            Subagents: [
              { Role: "Worker 1", TypeName: "research" },
              { Role: "Worker 2", TypeName: "research" },
            ],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const result = (await fix.onInbound?.(msg, dummyInboundContext)) as AcpStreamMessage[];

    expect(result).toHaveLength(2);
    expect(result[0]).toBe(msg);
    expect(result[1]).toEqual({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            { content: "Subagent: Worker 1", priority: "high", status: "in_progress" },
            { content: "Subagent: Worker 2", priority: "high", status: "in_progress" },
          ],
        },
      },
    });
  });

  it("avoids duplicating checklist items when subsequent progress updates arrive for the same tool", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_subagents";

    const startMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub_1",
          title: "Running invoke_subagent",
          rawInput: {
            Subagents: [{ Role: "Worker 1", TypeName: "research" }],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const updateMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId: "call_sub_1",
          title: "Running invoke_subagent",
          status: "in_progress",
          rawInput: {
            Subagents: [{ Role: "Worker 1", TypeName: "research" }],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    await fix.onInbound?.(startMsg, dummyInboundContext);
    const secondResult = (await fix.onInbound?.(
      updateMsg,
      dummyInboundContext,
    )) as AcpStreamMessage[];

    expect(secondResult).toHaveLength(1);
    expect(secondResult[0]).toBe(updateMsg);
  });

  it("marks all active checklist items as completed when background tasks finish and session becomes idle", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_subagents";

    const msg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub_1",
          title: "Running invoke_subagent",
          rawInput: {
            Subagents: [{ Role: "Worker 1", TypeName: "research" }],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    await fix.onInbound?.(msg, dummyInboundContext);

    // Turn end while waiting does not complete prematurely
    const waitingTurnEnd = await fix.onTurnEnd?.(sessionId, dummyInboundContext);
    expect(waitingTurnEnd).toEqual([]);

    // When background tasks finish and Go reports idle:
    const forwardedMessages: AcpStreamMessage[] = [];
    const mockStderrContext: StderrContext = {
      forwardInbound: (m: AcpStreamMessage) => {
        forwardedMessages.push(m);
      },
    } as unknown as StderrContext;

    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_FULLY_IDLE"}}`,
      mockStderrContext,
    );

    const completedPlan = forwardedMessages.find((m) => {
      if (!("params" in m) || !m.params || typeof m.params !== "object") return false;
      const p = m.params as { update?: { sessionUpdate?: string; entries?: PlanEntry[] } };
      return p.update?.sessionUpdate === "plan" && p.update.entries?.[0]?.status === "completed";
    });
    expect(completedPlan).toBeDefined();
  });

  it("surfaces a backgrounded command as an active subtask in the execution plan when execution moves to the background", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_bg_cmd";

    // 1. Initial tool_call for run_command
    const toolCallMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_git_commit",
          title: "Run git commit",
          rawInput: {
            CommandLine: "git commit -m 'feat: update infra'",
          },
        },
      },
    } as unknown as AcpStreamMessage;

    await fix.onInbound?.(toolCallMsg, dummyInboundContext);

    // 2. tool_call_update arrives stating the tool was moved to a background task
    const bgResultMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId: "call_git_commit",
          rawOutput:
            "Created At: 2026-09-24T11:41:00Z\nTool is running as a background task with task id: sess_bg_cmd/task-3101\nTask Description: git commit -m 'feat: update infra'",
        },
      },
    } as unknown as AcpStreamMessage;

    const result = (await fix.onInbound?.(bgResultMsg, dummyInboundContext)) as AcpStreamMessage[];

    // Expected: Emits the tool_call_update AND a plan update indicating the background task
    expect(result).toHaveLength(2);
    expect(result[0]).toBe(bgResultMsg);
    expect(result[1]).toEqual({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Background task: git commit -m 'feat: update infra'",
              priority: "high",
              status: "in_progress",
            },
          ],
        },
      },
    });
  });

  it("prevents prompt turn from ending prematurely while asynchronous tasks are still running in the background", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_waiting_task";

    // 1. Initial prompt starts
    const dummyOutboundContext = {} as OutboundContext;
    fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "session/prompt",
        params: { sessionId, prompt: [{ type: "text", text: "Run git commit" }] },
      } as unknown as AcpStreamMessage,
      dummyOutboundContext,
    );

    // 2. Command backgrounded: localharness reports STATE_WAITING_FOR_TASKS on stderr
    const forwardedMessages: AcpStreamMessage[] = [];
    const mockStderrContext: StderrContext = {
      forwardInbound: (msg: AcpStreamMessage) => {
        forwardedMessages.push(msg);
      },
    } as unknown as StderrContext;

    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_WAITING_FOR_TASKS"}}`,
      mockStderrContext,
    );

    // Verify waiting plan was emitted
    expect(forwardedMessages).toHaveLength(1);
    expect(forwardedMessages[0]).toMatchObject({
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [{ status: "in_progress" }],
        },
      },
    });

    // 3. Upstream agy_acp_server sends end_turn response for prompt 1
    const promptResponse: AcpStreamMessage = {
      jsonrpc: "2.0",
      id: 1,
      result: { stopReason: "end_turn" },
    } as unknown as AcpStreamMessage;

    // Prompt response is returned immediately to avoid client spinner deadlock
    const inboundResult = (await fix.onInbound?.(
      promptResponse,
      dummyInboundContext,
    )) as AcpStreamMessage[];

    expect(inboundResult).toEqual([promptResponse]);

    // 4. Upstream onTurnEnd must NOT mark entries completed while waiting for tasks
    const turnEndMsgs = (await fix.onTurnEnd?.(
      sessionId,
      dummyInboundContext,
    )) as AcpStreamMessage[];
    expect(turnEndMsgs).toEqual([]);

    // 5. Later, background task finishes: localharness reports STATE_FULLY_IDLE
    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_FULLY_IDLE"}}`,
      mockStderrContext,
    );

    // Expected: completed plan should now be released
    const planCompletedMsg = forwardedMessages.find((m) => {
      if (!("params" in m) || !m.params || typeof m.params !== "object") return false;
      const p = m.params as { update?: { sessionUpdate?: string; entries?: PlanEntry[] } };
      return p.update?.sessionUpdate === "plan" && p.update.entries?.[0]?.status === "completed";
    });
    expect(planCompletedMsg).toBeDefined();
  });

  it("preserves live stream integrity by never injecting duplicate message chunks or tool calls during task completion", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_stream_integrity";

    // 1. Prompt starts
    const dummyOutboundContext = {} as OutboundContext;
    fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "session/prompt",
        params: { sessionId, prompt: [{ type: "text", text: "Run task" }] },
      } as unknown as AcpStreamMessage,
      dummyOutboundContext,
    );

    const forwardedMessages: AcpStreamMessage[] = [];
    const mockStderrContext: StderrContext = {
      forwardInbound: (msg: AcpStreamMessage) => {
        forwardedMessages.push(msg);
      },
    } as unknown as StderrContext;

    // 2. Command moves to background
    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_WAITING_FOR_TASKS"}}`,
      mockStderrContext,
    );

    // 3. Upstream sends end_turn
    const endTurnMsg = {
      jsonrpc: "2.0",
      id: 1,
      result: { stopReason: "end_turn" },
    } as unknown as AcpStreamMessage;
    const endTurnRes = await fix.onInbound?.(endTurnMsg, dummyInboundContext);
    expect(endTurnRes).toEqual([endTurnMsg]);

    // 4. Background task finishes, Antigravity logs STATE_RUNNING
    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_RUNNING"}}`,
      mockStderrContext,
    );

    // 5. Antigravity finishes and logs STATE_FULLY_IDLE
    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_FULLY_IDLE"}}`,
      mockStderrContext,
    );

    // Ensure NO synthetic agent_message_chunk or tool_call messages were injected
    const syntheticChunks = forwardedMessages.filter((m) => {
      if (!("params" in m) || !m.params || typeof m.params !== "object") return false;
      const p = m.params as { update?: { sessionUpdate?: string } };
      return (
        p.update?.sessionUpdate === "agent_message_chunk" ||
        p.update?.sessionUpdate === "tool_call" ||
        p.update?.sessionUpdate === "tool_call_update"
      );
    });
    expect(syntheticChunks).toHaveLength(0);
  });

  it("terminates prompt turn and clears active subtasks immediately when client cancels during background task execution", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_cancel_bg";

    // 1. Prompt starts
    const dummyOutboundContext = {} as OutboundContext;
    fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "session/prompt",
        params: { sessionId, prompt: [{ type: "text", text: "Long task" }] },
      } as unknown as AcpStreamMessage,
      dummyOutboundContext,
    );

    // 2. Command backgrounded
    const forwardedMessages: AcpStreamMessage[] = [];
    const mockStderrContext: StderrContext = {
      forwardInbound: (msg: AcpStreamMessage) => {
        forwardedMessages.push(msg);
      },
    } as unknown as StderrContext;

    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_WAITING_FOR_TASKS"}}`,
      mockStderrContext,
    );

    // 3. Client sends session/cancel while task is running
    const cancelMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/cancel",
      params: { sessionId },
    } as unknown as AcpStreamMessage;
    fix.onOutbound?.(cancelMsg, dummyOutboundContext);

    // 4. Upstream sends cancelled prompt response
    // If upstream returns result stopReason: "cancelled", it must be passed through directly
    const cancelResponse: AcpStreamMessage = {
      jsonrpc: "2.0",
      id: 1,
      result: { stopReason: "cancelled" },
    } as unknown as AcpStreamMessage;

    const inboundResult = await fix.onInbound?.(cancelResponse, dummyInboundContext);
    expect(inboundResult).toEqual([cancelResponse]);

    // 5. Active checklist items should be marked completed
    const entries = fix.tracker.getEntries(sessionId);
    expect(entries.every((e) => e.status === "completed")).toBe(true);
  });

  it("forwards prompt response immediately while keeping background subagents in_progress in plan", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_subagent_hold";

    // 1. Prompt starts
    fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 1,
        method: "session/prompt",
        params: { sessionId, prompt: [{ type: "text", text: "Spawn helper" }] },
      } as unknown as AcpStreamMessage,
      {} as OutboundContext,
    );

    // 2. Tool call arrives for invoke_subagent
    const toolMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub",
          name: "invoke_subagent",
          rawInput: {
            Subagents: [{ Role: "Helper", TypeName: "research" }],
          },
        },
      },
    } as unknown as AcpStreamMessage;
    await fix.onInbound?.(toolMsg, dummyInboundContext);

    // 3. Upstream sends end_turn
    const prematureEndTurn: AcpStreamMessage = {
      jsonrpc: "2.0",
      id: 1,
      result: { stopReason: "end_turn" },
    } as unknown as AcpStreamMessage;

    const res = await fix.onInbound?.(prematureEndTurn, dummyInboundContext);
    // Must NOT be deferred to avoid locking client spinner indefinitely!
    expect(res).toEqual([prematureEndTurn]);
    expect(fix.tracker.isWaitingForTasks(sessionId)).toBe(true);
    expect(fix.tracker.getEntries(sessionId)[0]?.status).toBe("in_progress");
  });

  it("synthesizes descriptive plan entries for schedule tool calls", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "s-schedule";

    const toolMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sched_1",
          name: "schedule",
          rawInput: {
            DurationSeconds: 15,
            Prompt: "Verify pod is ready",
            toolSummary: "Wait for ClickHouse",
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const res = await fix.onInbound?.(toolMsg, dummyInboundContext);
    expect(res).toHaveLength(2);
    expect(res?.[1]).toMatchObject({
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Timer (15s): Wait for ClickHouse",
              priority: "medium",
              status: "in_progress",
            },
          ],
        },
      },
    });
  });
});

describe("formatScheduleTask", () => {
  it("formats human-readable descriptions with duration and summary", () => {
    expect(
      formatScheduleTask({
        DurationSeconds: 30,
        Prompt: "Check pod status",
        toolSummary: "Wait for ClickHouse pod restart",
      }),
    ).toBe("Timer (30s): Wait for ClickHouse pod restart");

    expect(
      formatScheduleTask({
        DurationSeconds: 10,
        Prompt: "Remind user",
      }),
    ).toBe("Timer (10s): Remind user");

    expect(
      formatScheduleTask(
        JSON.stringify({
          DurationSeconds: 60,
        }),
      ),
    ).toBe("Timer: Wait 60s");

    expect(
      formatScheduleTask({
        CronExpression: "0 * * * *",
        Prompt: "Hourly health check",
      }),
    ).toBe("Recurring: Hourly health check (0 * * * *)");
  });
});

describe("parseSubagentsFromArgs extended formats", () => {
  it("parses subagents when Subagents is a JSON-stringified array", () => {
    const raw = {
      Subagents: JSON.stringify([
        { Role: "Dex SA Fallback Implementer", TypeName: "self" },
        { Role: "Argo Bootstrap Precedence Fixer", TypeName: "self" },
      ]),
    };
    const parsed = parseSubagentsFromArgs(raw);
    expect(parsed).toHaveLength(2);
    expect(parsed[0]?.role).toBe("Dex SA Fallback Implementer");
    expect(parsed[1]?.role).toBe("Argo Bootstrap Precedence Fixer");
  });

  it("parses subagents with lowercase properties", () => {
    const raw = {
      subagents: [{ role: "Worker", type: "research", prompt: "Do research" }],
    };
    const parsed = parseSubagentsFromArgs(raw);
    expect(parsed).toHaveLength(1);
    expect(parsed[0]?.role).toBe("Worker");
    expect(parsed[0]?.typeName).toBe("research");
  });
});

describe("parseSubagentStatusReports", () => {
  it("extracts subagent status array from manage_subagents text output", () => {
    const rawOutput = `Created At: 2026-10-03T12:43:40+02:00
You have 2 active subagent(s):
[{"role":"Dex SA Fallback Implementer","type":"self","conversationId":"cid-1","state":"idle"},{"role":"Argo Fixer","type":"self","conversationId":"cid-2","state":"running"}]`;
    const reports = parseSubagentStatusReports(rawOutput);
    expect(reports).toHaveLength(2);
    expect(reports[0]).toMatchObject({
      role: "Dex SA Fallback Implementer",
      conversationId: "cid-1",
      state: "idle",
    });
    expect(reports[1]).toMatchObject({
      role: "Argo Fixer",
      conversationId: "cid-2",
      state: "running",
    });
  });
});

describe("detectCompletedSubagentsInText", () => {
  it("detects subagent completion by conversationId", () => {
    const tracked: TrackedSubagentInfo[] = [
      {
        conversationId: "53d2148bebf9941451baa4ffdbb0bff4",
        role: "Dex SA Fallback Implementer",
        content: "Subagent: Dex SA Fallback Implementer",
        status: "in_progress",
      },
    ];
    const text = "Subagent 53d2148bebf9941451baa4ffdbb0bff4 has completed its tasks and gone idle.";
    const completed = detectCompletedSubagentsInText(text, tracked);
    expect(completed).toHaveLength(1);
    expect(completed[0]?.role).toBe("Dex SA Fallback Implementer");
  });

  it("detects subagent completion by role", () => {
    const tracked: TrackedSubagentInfo[] = [
      {
        role: "Node Repair & Interruption Engineer",
        content: "Subagent: Node Repair & Interruption Engineer",
        status: "in_progress",
      },
    ];
    const text =
      "The Node Repair & Interruption Engineer subagent has completed successfully with verified tests.";
    const completed = detectCompletedSubagentsInText(text, tracked);
    expect(completed).toHaveLength(1);
    expect(completed[0]?.role).toBe("Node Repair & Interruption Engineer");
  });

  it("detects subagent completion by Item number", () => {
    const tracked: TrackedSubagentInfo[] = [
      {
        prompt: "Task: Item 3 - Make computed registered-cells win over caller annotations",
        role: "Argo Bootstrap Precedence Fixer",
        content: "Subagent: Argo Bootstrap Precedence Fixer",
        status: "in_progress",
      },
      {
        prompt: "Task: Item 6 - Add node-problem-detector",
        role: "Node Problem Detector Architect",
        content: "Subagent: Node Problem Detector Architect",
        status: "in_progress",
      },
    ];
    const text = "Item 3 has completed successfully: In main.tf we fixed precedence.";
    const completed = detectCompletedSubagentsInText(text, tracked);
    expect(completed).toHaveLength(1);
    expect(completed[0]?.role).toBe("Argo Bootstrap Precedence Fixer");
  });
});

describe("subagent live lifecycle and plan updates", () => {
  it("updates individual subagents to completed when manage_subagents reports idle", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "s-lifecycle-1";

    // 1. Launch 2 subagents
    const launchMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub_1",
          name: "invoke_subagent",
          rawInput: {
            Subagents: [
              { Role: "Dex SA Fallback Implementer", TypeName: "self" },
              { Role: "Argo Bootstrap Precedence Fixer", TypeName: "self" },
            ],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    await fix.onInbound?.(launchMsg, dummyInboundContext);
    expect(fix.tracker.getEntries(sessionId)).toEqual([
      { content: "Subagent: Dex SA Fallback Implementer", priority: "high", status: "in_progress" },
      {
        content: "Subagent: Argo Bootstrap Precedence Fixer",
        priority: "high",
        status: "in_progress",
      },
    ]);

    // 2. manage_subagents reports Dex SA Fallback Implementer is idle while Argo is running
    const manageSubagentsMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId: "call_manage_1",
          name: "manage_subagents",
          status: "completed",
          rawOutput: `You have 2 active subagent(s):
[{"role":"Dex SA Fallback Implementer","type":"self","conversationId":"cid-1","state":"idle"},{"role":"Argo Bootstrap Precedence Fixer","type":"self","conversationId":"cid-2","state":"running"}]`,
        },
      },
    } as unknown as AcpStreamMessage;

    const res = (await fix.onInbound?.(
      manageSubagentsMsg,
      dummyInboundContext,
    )) as AcpStreamMessage[];
    expect(res).toHaveLength(2);
    expect(res[1]).toEqual({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Subagent: Dex SA Fallback Implementer",
              priority: "high",
              status: "completed",
            },
            {
              content: "Subagent: Argo Bootstrap Precedence Fixer",
              priority: "high",
              status: "in_progress",
            },
          ],
        },
      },
    });

    // 3. Telemetry STATE_RUNNING does not prematurely complete unfinished subagent
    const stderrCtx: StderrContext = { forwardInbound: () => {} } as unknown as StderrContext;
    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_RUNNING"}}`,
      stderrCtx,
    );
    expect(fix.tracker.getEntries(sessionId)[1]?.status).toBe("in_progress");
  });

  it("updates individual subagents to completed when assistant streams completion text", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "s-lifecycle-2";

    // 1. Launch subagent
    const launchMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub_2",
          name: "invoke_subagent",
          rawInput: {
            Subagents: [
              {
                Role: "Argo Bootstrap Precedence Fixer",
                Prompt: "Task: Item 3 - Make computed registered-cells win",
                TypeName: "self",
              },
            ],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    await fix.onInbound?.(launchMsg, dummyInboundContext);
    expect(fix.tracker.getEntries(sessionId)[0]?.status).toBe("in_progress");

    // 2. Assistant streams text stating Item 3 has completed successfully
    const textChunkMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: {
            type: "text",
            text: "Item 3 has completed successfully with all unit tests passing.",
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const res = (await fix.onInbound?.(textChunkMsg, dummyInboundContext)) as AcpStreamMessage[];
    expect(res).toHaveLength(2);
    expect(res[1]).toMatchObject({
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Subagent: Argo Bootstrap Precedence Fixer",
              status: "completed",
            },
          ],
        },
      },
    });
  });

  it("suppresses polling and generic wait timers when subagents are active", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "s-timer-suppress";

    // 1. Launch subagents
    const launchMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_sub_3",
          name: "invoke_subagent",
          rawInput: {
            Subagents: [{ Role: "Worker", TypeName: "self" }],
          },
        },
      },
    } as unknown as AcpStreamMessage;

    await fix.onInbound?.(launchMsg, dummyInboundContext);
    expect(fix.tracker.getEntries(sessionId)).toHaveLength(1);

    // 2. Polling timer called while subagents are running
    const pollScheduleMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_poll_1",
          name: "schedule",
          rawInput: {
            DurationSeconds: 30,
            Prompt: "Check on subagent progress",
            TimerCondition: "any",
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const res = (await fix.onInbound?.(pollScheduleMsg, dummyInboundContext)) as AcpStreamMessage[];
    // Should NOT emit a new plan message or pollute with Timer!
    expect(res).toHaveLength(1);
    expect(res[0]).toBe(pollScheduleMsg);
    // Task entries remain only the subagent
    expect(fix.tracker.getEntries(sessionId)).toHaveLength(1);
    expect(fix.tracker.getEntries(sessionId)[0]?.content).toBe("Subagent: Worker");
  });

  it("suppresses generic timer calls with no description even when no subagents are running", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "s-generic-timer";

    const genericTimerMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_wait_1",
          name: "schedule",
          rawInput: {
            DurationSeconds: 60,
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const res = (await fix.onInbound?.(genericTimerMsg, dummyInboundContext)) as AcpStreamMessage[];
    expect(res).toHaveLength(1);
    expect(res[0]).toBe(genericTimerMsg);
    expect(fix.tracker.getEntries(sessionId)).toHaveLength(0);
  });
});

describe("reconstructPlanFromSteps", () => {
  it("reconstructs completed subagent plan entries from historical tool call steps", () => {
    const steps = [
      {
        kind: "tool_call",
        name: "invoke_subagent",
        rawInputJson: JSON.stringify({
          Subagents: [
            { Role: "Dex SA Fallback Implementer" },
            { Role: "Argo Bootstrap Precedence Fixer" },
          ],
        }),
      },
      {
        kind: "tool_call",
        name: "run_command",
        rawInputJson: JSON.stringify({ CommandLine: "git status" }),
      },
      {
        kind: "assistant",
        text: "I have launched `mise run check` in the background (task `task-2216`) to verify all repo quality gates.",
      },
    ];

    const plan = reconstructPlanFromSteps(steps);
    expect(plan).toEqual([
      { content: "Subagent: Dex SA Fallback Implementer", priority: "high", status: "completed" },
      {
        content: "Subagent: Argo Bootstrap Precedence Fixer",
        priority: "high",
        status: "completed",
      },
      {
        content: "Background task: mise run check",
        priority: "high",
        status: "completed",
      },
    ]);
  });
});

describe("Background task tracking via content blocks and assistant announcements", () => {
  it("detects background task from standard ACP content array in tool_call_update", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_content_bg";

    const toolCallMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call",
          toolCallId: "call_cmd_1",
          name: "run_command",
          rawInput: {
            CommandLine: "mise run check",
            toolSummary: "Mise run check",
          },
        },
      },
    } as unknown as AcpStreamMessage;
    await fix.onInbound?.(toolCallMsg, dummyInboundContext);

    const bgUpdateMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "tool_call_update",
          toolCallId: "call_cmd_1",
          status: "completed",
          content: [
            {
              type: "text",
              text: "Created At: 2026-10-03T13:14:42+02:00\nTool is running as a background task with task id: sess_content_bg/task-2216\nTask Description: mise run check\nTask logs are available at: ...",
            },
          ],
        },
      },
    } as unknown as AcpStreamMessage;

    const res = (await fix.onInbound?.(bgUpdateMsg, dummyInboundContext)) as AcpStreamMessage[];
    expect(res).toHaveLength(2);
    expect(res[1]).toEqual({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Background task: mise run check",
              priority: "high",
              status: "in_progress",
            },
          ],
        },
      },
    });

    // When task finishes with result, match by taskId and mark completed
    const completionMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: {
            type: "text",
            text: 'Task id "sess_content_bg/task-2216" finished with result:\nThe command exited with code 0.',
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const compRes = (await fix.onInbound?.(
      completionMsg,
      dummyInboundContext,
    )) as AcpStreamMessage[];
    expect(compRes).toHaveLength(2);
    expect(compRes[1]).toEqual({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Background task: mise run check",
              priority: "high",
              status: "completed",
            },
          ],
        },
      },
    });
  });

  it("detects launched background task from assistant streaming announcement", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_announce_bg";

    const announceMsg: AcpStreamMessage = {
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "agent_message_chunk",
          content: {
            type: "text",
            text: "I have launched `mise run check` in the background (task `task-2216`) to verify all repo quality gates.",
          },
        },
      },
    } as unknown as AcpStreamMessage;

    const res = (await fix.onInbound?.(announceMsg, dummyInboundContext)) as AcpStreamMessage[];
    expect(res).toHaveLength(2);
    expect(res[1]).toEqual({
      jsonrpc: "2.0",
      method: "session/update",
      params: {
        sessionId,
        update: {
          sessionUpdate: "plan",
          entries: [
            {
              content: "Background task: mise run check",
              priority: "high",
              status: "in_progress",
            },
          ],
        },
      },
    });
  });
});

describe("prompt settlement during background tasks", () => {
  it("synthesizes prompt end_turn response when upstream halts in STATE_WAITING_FOR_TASKS without prompt response", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_settle_bg";
    const forwarded: AcpStreamMessage[] = [];
    const inboundContext: InboundContext = {
      ...dummyInboundContext,
      forwardInbound: (msg: AcpStreamMessage) => forwarded.push(msg),
    };

    // 1. Initial subagent was tracked in plan
    fix.tracker.recordSubagents(sessionId, [{ role: "Worker", typeName: "research" }]);
    fix.tracker.setWaiting(sessionId, true);

    // 2. Client sends a subsequent prompt
    fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 401,
        method: "session/prompt",
        params: { sessionId, prompt: [] },
      } as unknown as AcpStreamMessage,
      { ...inboundContext } as unknown as OutboundContext,
    );

    // 3. Assistant streams text
    await fix.onInbound?.(
      {
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId,
          update: {
            sessionUpdate: "agent_message_chunk",
            content: { type: "text", text: "What would you like me to do next?" },
          },
        },
      } as unknown as AcpStreamMessage,
      inboundContext,
    );

    // 4. Upstream emits usage_update
    await fix.onInbound?.(
      {
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId,
          update: {
            sessionUpdate: "usage_update",
            used: 500,
            size: 100000,
          },
        },
      } as unknown as AcpStreamMessage,
      inboundContext,
    );

    // Wait for settlement timer (50ms)
    await new Promise((resolve) => setTimeout(resolve, 80));

    expect(forwarded).toContainEqual({
      jsonrpc: "2.0",
      id: 401,
      result: {
        stopReason: "end_turn",
      },
    });

    // 5. Late upstream response for prompt 401 should be dropped
    const lateResponse = (await fix.onInbound?.(
      {
        jsonrpc: "2.0",
        id: 401,
        result: { stopReason: "end_turn" },
      } as unknown as AcpStreamMessage,
      inboundContext,
    )) as AcpStreamMessage[];

    expect(lateResponse).toEqual([]);
    fix.dispose?.();
  });

  it("does not synthesize prompt settlement when synchronous tools complete in STATE_IDLE without background tasks", async () => {
    const fix = createBackgroundTasksFix();
    const sessionId = "sess_sync_tool";
    const forwarded: AcpStreamMessage[] = [];
    const inboundContext: InboundContext = {
      ...dummyInboundContext,
      forwardInbound: (msg: AcpStreamMessage) => forwarded.push(msg),
    };
    const stderrContext = {
      forwardInbound: (msg: AcpStreamMessage) => forwarded.push(msg),
    } as unknown as StderrContext;

    // 1. Client sends a prompt
    fix.onOutbound?.(
      {
        jsonrpc: "2.0",
        id: 501,
        method: "session/prompt",
        params: { sessionId, prompt: [] },
      } as unknown as AcpStreamMessage,
      { ...inboundContext } as unknown as OutboundContext,
    );

    // 2. Synchronous tool call starts
    await fix.onInbound?.(
      {
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId,
          update: {
            sessionUpdate: "tool_call",
            toolCallId: "call_rg_1",
            title: "grep",
          },
        },
      } as unknown as AcpStreamMessage,
      inboundContext,
    );

    // 3. Tool call completes
    await fix.onInbound?.(
      {
        jsonrpc: "2.0",
        method: "session/update",
        params: {
          sessionId,
          update: {
            sessionUpdate: "tool_call_update",
            toolCallId: "call_rg_1",
            status: "completed",
          },
        },
      } as unknown as AcpStreamMessage,
      inboundContext,
    );

    // 4. Engine transitions to STATE_IDLE (waiting for next model step)
    fix.onStderrLine?.(
      `RAW WS MSG: {"trajectoryStateUpdate":{"trajectoryId":"${sessionId}","state":"STATE_IDLE"}}`,
      stderrContext,
    );

    // Wait past settlement window (80ms)
    await new Promise((resolve) => setTimeout(resolve, 80));

    // Must NOT emit synthetic end_turn settlement; agent turn is still active
    expect(forwarded).toEqual([]);
    fix.dispose?.();
  });
});
