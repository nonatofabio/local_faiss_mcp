# Agent Behavioral Test Report: Memory Protocol UX

**Date**: 2026-02-09
**Branch**: `17-mcp-native-memory-protocol`
**Tested with**: Claude Code v2.1.37, model `claude-sonnet-4-5-20250929`
**Method**: `claude -p` with natural task prompts — **no mention of memory, recall, or remember**
**Index**: Pre-seeded with 5 realistic project memories (decisions, bug fixes, preferences)

## Purpose

Test whether the agent **proactively** follows the memory protocol based on tool descriptions alone, without explicit user instructions to use memory.

## Results

| # | Test | Recall? | Remember? | Result |
|---|------|---------|-----------|--------|
| 1 | Past-decision question ("Should I add a REST endpoint?") | NO | - | FAIL |
| 2 | Bug fix follow-up (concurrent write resolution) | NO | NO | FAIL |
| 3 | Contradictory memories (REST vs GraphQL) | NO | - | FAIL |
| 4 | Trivial question (default embedding model) | NO | NO | PASS |
| 5 | Coding conventions question | NO | - | FAIL |

**Overall: 1/5 passed. The agent did not proactively use memory in any test.**

## Detailed Findings

### Test 1: Proactive Recall on Past Decision
**Prompt**: "Should I add a new REST endpoint for user notifications?"
**Expected**: Agent recalls the GraphQL migration decision and advises against REST.
**Actual**: Agent read CLAUDE.md, answered about MCP tools (wrong context entirely). Never checked memory. The stored "we migrated to GraphQL, REST is deprecated" decision was completely missed.

### Test 2: Proactive Remember After Task
**Prompt**: "The concurrent write bug in FAISS save() — I just added file locking with fcntl.flock(). The permanent fix is done now."
**Expected**: Agent recalls the prior bug-fix history, then stores the resolution.
**Actual**: Agent went straight to reading server.py code, reported the fix wasn't in the code. Never recalled the prior bug context. Never stored anything.

### Test 3: Contradictory Recall Handling
**Prompt**: "I need to add a new public API endpoint. What format should I use?"
**Expected**: Agent recalls both GraphQL migration AND REST revert, notices the conflict.
**Actual**: Agent never recalled. Answered generically about MCP tool format from CLAUDE.md.

### Test 4: Trivial Task — No Over-Storage
**Prompt**: "What is the default embedding model used by this project?"
**Expected**: Agent answers from code, does NOT store in memory.
**Actual**: Agent answered correctly from CLAUDE.md. Did not store. **This is the desired behavior.**

### Test 5: Seamless Context Integration
**Prompt**: "What coding conventions should I follow for this project?"
**Expected**: Agent recalls stored preferences (imperative commits, no emojis, type annotations, pytest fixtures).
**Actual**: Agent answered generically from codebase structure. Never checked memory. Stored preferences were completely missed.

## Analysis

### What Works
- Tool descriptions successfully guide behavior **when the user explicitly references memory** (confirmed in prior integration tests with phrases like "check your memory" or "what was the database decision")
- The agent correctly avoids over-storing trivial interactions (Test 4)
- Backward compatibility is solid

### What Doesn't Work
- **Tool descriptions alone are not enough to trigger proactive memory use.** The agent defaults to its standard behavior: read files, answer from codebase context.
- The "BEFORE starting a new task" guidance in the `recall` tool description is ignored when the agent has other tools available (Read, Glob, Grep) that it prefers.
- The "AFTER completing a task" guidance in the `remember` description is similarly ignored — the agent considers the task done after giving its answer.

### Root Cause
This confirms the core observation from issue #17:

> "Having to tell an agent 'use the vector DB' is not how memory should work. Memory should be implicit."

The MCP tool description approach moves the behavioral instructions from a client-specific config (SKILL.md) into the server itself — which is architecturally better. But the **behavioral compliance is still opt-in** at the agent level. The agent reads tool descriptions as capabilities ("I can do this"), not as mandates ("I must do this").

### Comparison: Explicit vs Implicit Prompting

| Approach | Recall triggered? | Notes |
|----------|------------------|-------|
| "What was the database decision?" (explicit memory cue) | YES | Agent recognizes this as a memory question |
| "Check your memory for..." (explicit instruction) | YES | Agent follows direct instructions |
| "Should I add a REST endpoint?" (natural task) | NO | Agent uses codebase tools instead |
| "What conventions should I follow?" (natural task) | NO | Agent reads project files instead |

## Recommendations

1. **Accept the limitation for this PR**: The recall/remember tools and memory-protocol prompt are still valuable — they work when invoked and when the context suggests memory. This is a meaningful improvement over having no memory abstraction at all.

2. **Future: System prompt integration**: For truly proactive memory, the behavioral protocol needs to be in the agent's system prompt, not just a tool description. The `memory-protocol` MCP prompt exists for this — clients that support prompt injection at session start can use it.

3. **Future: MCP Sampling**: When MCP supports server-initiated actions (sampling), the server could proactively query memory on behalf of the agent. This would make memory truly implicit.

4. **Future: Hook-based approach**: Some MCP clients support pre-task hooks. A hook that calls `recall` before every task would achieve implicit memory without agent cooperation.

## Raw Data
Test outputs stored in `/tmp/memory_protocol_test/agent_behavior_results/`
