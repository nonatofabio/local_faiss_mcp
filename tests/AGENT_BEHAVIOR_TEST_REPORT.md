# Agent Behavioral Test Report: Memory Protocol UX

**Date**: 2026-02-09
**Branch**: `17-mcp-native-memory-protocol`
**Tested with**: Claude Code v2.1.37, model `claude-sonnet-4-5-20250929`
**Method**: `claude -p` with natural task prompts — **no mention of memory, recall, or remember**
**Index**: Pre-seeded with 5 realistic project memories (decisions, bug fixes, preferences)

## Purpose

Test whether the agent **proactively** follows the memory protocol based on tool descriptions alone, without explicit user instructions to use memory.

## Round 1: Tool Descriptions Only (run from project directory)

| # | Test | Recall? | Remember? | Result |
|---|------|---------|-----------|--------|
| 1 | Past-decision question ("Should I add a REST endpoint?") | NO | - | FAIL |
| 2 | Bug fix follow-up (concurrent write resolution) | NO | NO | FAIL |
| 3 | Contradictory memories (REST vs GraphQL) | NO | - | FAIL |
| 4 | Trivial question (default embedding model) | NO | NO | PASS |
| 5 | Coding conventions question | NO | - | FAIL |

**Overall: 1/5 passed. The agent did not proactively use memory in any test.**

### What Happened
- Agent defaulted to reading CLAUDE.md and source files every time
- Tool descriptions ("use BEFORE starting a task") were treated as suggestions, not mandates
- When the codebase had enough context to answer, the agent never reached for memory

## Round 2: Server Naming (no effect)

Renamed the MCP server from `local-faiss-mcp` to `memory` so tools appear as `mcp__memory__recall`. Reran tests 1, 2, 3.

**Result: 0/3. No change.** Tool names alone don't influence agent prioritization.

## Round 3: Server `instructions` Field (run from project directory)

Added behavioral instructions to the MCP `Server()` constructor:

> "This server is your long-term memory. Use the 'recall' tool BEFORE starting any task..."

Reran tests 1, 2, 3 from the **project directory** (where CLAUDE.md exists).

**Result: 0/3. No change.** When CLAUDE.md is available, the agent still prefers local file reading over memory tools, even with server instructions.

## Round 4: Server `instructions` + Outside Project Directory

Same server instructions, but ran `claude -p` from `/tmp/memory_protocol_test/` — a directory with **no CLAUDE.md, no source code, no codebase context**.

| # | Test | Recall? | Context Integrated? | Result |
|---|------|---------|---------------------|--------|
| 1 | "Should I add a new REST endpoint?" | YES | YES — cited GraphQL→REST revert | PASS |
| 2 | Bug fix follow-up | YES | YES — recalled prior bug history | PASS |
| 3 | "What coding conventions?" | YES | YES — surfaced stored preferences | PASS |

**Overall: 3/3 passed.**

### What Changed
- The agent called `recall` as its **first action** in all three tests
- Retrieved context was seamlessly integrated into responses
- Test 1: Agent found the REST revert decision and recommended REST for the new endpoint
- Test 3: Agent surfaced "no emojis, imperative commits, type annotations, pytest fixtures" — all from memory

## Analysis

### The Two Factors

Proactive memory use requires **both**:

1. **Server `instructions`** — tells the agent "this is your memory, use it before every task"
2. **Absence of richer local context** — when CLAUDE.md and source files are available, the agent satisfies the query from those and never reaches for memory

This means the MCP server instructions **work** — but they compete with the agent's preference for local file reading. Memory becomes the fallback, not the first choice.

### When Memory Wins vs When Files Win

| Scenario | Agent uses memory? | Why |
|----------|-------------------|-----|
| No local codebase context | YES | Memory is the only source of project knowledge |
| CLAUDE.md exists with relevant info | NO | Agent satisfies query from local files |
| User explicitly references past decisions | YES | Agent recognizes this needs historical context |
| Question about conventions/preferences (not in files) | DEPENDS | Only if that info isn't in CLAUDE.md |

### Implications for Real Usage

In practice, memory is most valuable for information that **isn't in the codebase**:
- Past decisions and their rationale
- Bug fix history and lessons learned
- User preferences not documented in CLAUDE.md
- Cross-session context

The server `instructions` field ensures the agent checks memory when it doesn't have local files to lean on — which is exactly the scenario where memory matters most (new sessions, new directories, cross-project context).

## Recommendations

1. **Ship with server `instructions`** — they work and they're MCP-standard
2. **Document the interaction with CLAUDE.md** — users should know that memory complements, not replaces, local project files
3. **Future: System prompt injection** — for environments where memory must override local files, the `memory-protocol` MCP prompt can be injected into the agent's system prompt by supporting clients
4. **Future: MCP Sampling / Hooks** — server-initiated recall would bypass the agent's tool prioritization entirely
