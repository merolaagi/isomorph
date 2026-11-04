"""Claude as a weight-space programmer: a tool-use loop over a Workspace.

The loop sends the task and the tool list, runs every tool Claude asks for
against the workspace, and returns the results, until Claude calls `finish`
or the turn limit is reached. Claude never runs code: every action is one of
the workspace's tools, and every call and result is kept in the transcript.

`call` (one request to the Messages API) can be replaced, which is how the
tests drive the loop without a network or a key.
"""
import json
import time
import urllib.error
import urllib.request

from .workspace import render

AGENT_MODEL = "claude-opus-5-5"
API = "https://api.anthropic.com/v1/messages"

SYSTEM = """You are a weight-space programmer. You read, and when allowed edit, the weights of a neural network directly, with the tools you are given. There is no training loop and no gradient: you work out what parts of the network compute from their weights, and you change weights by reasoning about what they do.

How to work:
- Start broad (list_tensors, tensor_stats), then follow the structure you find (svd, project, read_direction, fft, compare).
- Directions are passed between tools by id (d1, d2, ...). Chain them: a singular vector of one matrix, pushed through the next, read out as tokens.
- Forward passes are scarce when they are offered at all. Plan each one: write texts that separate the hypotheses you hold.
- Edits are undoable. Keep an edit only if the practice score improves; undo the ones that do not. Small, targeted edits are better than large ones.
- When you are done, call finish once. Say what you found and how sure you are, and separate what the weights showed from what you guess.
"""


def finish_tool(schema):
    return {"name": "finish", "description": "End the task with your answer. Call it exactly once.",
            "input_schema": {"type": "object", "properties": schema, "required": list(schema)}}


def api_call(key, model=AGENT_MODEL, max_tokens=6000):
    def call(system, messages, tools):
        body = {"model": model, "max_tokens": max_tokens, "system": system, "messages": messages, "tools": tools}
        for attempt in range(5):
            req = urllib.request.Request(API, data=json.dumps(body).encode(),
                                         headers={"content-type": "application/json", "x-api-key": key, "anthropic-version": "2023-06-01"})
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return json.loads(r.read())
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="ignore")[:400]
                if e.code in (429, 500, 502, 503, 529) and attempt < 4:
                    time.sleep(4 * (attempt + 1))
                    continue
                raise RuntimeError(f"Anthropic API error {e.code}: {detail}")
        raise RuntimeError("The Anthropic API did not answer.")
    return call


def run_agent(ws, task_text, finish_schema, call, max_turns=40, progress=lambda f, m: None, should_stop=lambda: False,
              on_tool=None):
    """Drive the loop. Returns (finish arguments or None, transcript, usage)."""
    tools = [{k: t[k] for k in ("name", "description", "input_schema")} for t in ws.tool_table()]
    tools.append(finish_tool(finish_schema))
    messages = [{"role": "user", "content": task_text}]
    transcript, usage = [], {"input_tokens": 0, "output_tokens": 0, "turns": 0}
    answer, nudged = None, False
    for turn in range(max_turns):
        if should_stop():
            raise InterruptedError("Stopped.")
        progress(turn / max_turns, f"Turn {turn + 1}: {len(ws.calls)} tool calls, {ws.forward_used} of {ws.forward_budget} forward passes")
        resp = call(SYSTEM, messages, tools)
        u = resp.get("usage") or {}
        usage["input_tokens"] += int(u.get("input_tokens", 0))
        usage["output_tokens"] += int(u.get("output_tokens", 0))
        usage["turns"] += 1
        content = resp.get("content") or []
        messages.append({"role": "assistant", "content": content})
        results = []
        for b in content:
            if b.get("type") == "text" and b.get("text", "").strip():
                transcript.append({"kind": "thought", "text": b["text"]})
            elif b.get("type") == "tool_use":
                name, args = b["name"], b.get("input") or {}
                if name == "finish":
                    answer = args
                    transcript.append({"kind": "finish", "args": args})
                    results.append({"type": "tool_result", "tool_use_id": b["id"], "content": "Recorded."})
                    continue
                res = ws.call(name, args)
                transcript.append({"kind": "tool", "tool": name, "args": args, "result": res})
                if on_tool:
                    on_tool(name, args, res)
                results.append({"type": "tool_result", "tool_use_id": b["id"], "content": render(res),
                                **({"is_error": True} if "error" in res else {})})
        if answer is not None:
            break
        if results:
            messages.append({"role": "user", "content": results})
        else:
            if nudged:
                break
            nudged = True
            messages.append({"role": "user", "content": "Call finish now with your answer."})
    return answer, transcript, usage
