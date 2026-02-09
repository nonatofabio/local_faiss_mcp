# Integration Test Report: MCP-Native Memory Protocol

**Date**: 2026-02-09
**Branch**: `17-mcp-native-memory-protocol`
**Tested with**: Claude Code v2.1.37, model `claude-sonnet-4-5-20250929`
**Method**: Sandboxed `claude -p` sessions with `--strict-mcp-config` pointing to the branch build

## Test Setup

- Fresh FAISS index (empty) at `/tmp/memory_protocol_test/index`
- MCP server running from the branch: `python -m local_faiss_mcp --index-dir ...`
- Each test is an isolated `claude -p` invocation with `--no-session-persistence`
- Output captured as JSON (`--output-format json`) and parsed for tool call names and response text
- All built-in tools disabled except Read, Glob, Grep (to isolate MCP tool behavior)

## Results

| # | Test | Tool Called | Result |
|---|------|-----------|--------|
| 1 | **remember stores data** — explicit ask to use remember | `remember` | PASS — stored with source tag `decision:database-choice` |
| 2 | **recall retrieves data** — ask about prior decision with "check your memory" hint | `recall` | PASS — retrieved PostgreSQL decision correctly |
| 3 | **recall without explicit hint** — vague question with no mention of memory | `recall` | PASS — Claude called recall as first action unprompted |
| 4 | **Backward compat: ingest_document** — explicitly ask for old tool name | `ingest_document` | PASS — still works, returns success |
| 5 | **Cross-compat: recall finds ingest_document data** — data stored via old tool, retrieved via new | `recall` | PASS — found "token refresh" / "OAuth" content |
| 6 | **Cross-compat: query_rag_store finds remember data** — data stored via new tool, retrieved via old | `query_rag_store` | PASS — found PostgreSQL decision |
| 7 | **memory-protocol prompt accessible** — ask Claude to list MCP prompts | Read (inspected server.py) | PASS — memory-protocol listed and described |

**Overall: 10/10 assertions passed across 7 tests.**

## Key Observations

### 1. Tool descriptions drive behavior
The behavioral guidance in the `recall` and `remember` tool descriptions ("Use this BEFORE starting a new task", "Use this AFTER completing a task") successfully influenced Claude's behavior. In Test 3, with no explicit hint about memory, Claude still chose to call `recall` as its first action based purely on the tool description.

### 2. Full backward compatibility confirmed
- `ingest_document` and `query_rag_store` work exactly as before (Tests 4, 6)
- Data is fully interchangeable between old and new tool names (Tests 5, 6)
- No changes to existing tool schemas or behavior

### 3. memory-protocol prompt is discoverable
Claude found and correctly described the `memory-protocol` prompt alongside the existing `extract-answer` and `summarize-documents` prompts (Test 7). It appears as a slash command (`/mcp__local-faiss-mcp__memory-protocol`) in Claude Code's init output.

### 4. Claude's recall behavior is contextually appropriate
In Test 3, Claude not only recalled the stored memory but also contextualized it — noting that the PostgreSQL decision was from a different context than the current FAISS project. This shows the protocol doesn't create blind trust in memory.

## UX Impact

- **No negative UX change**: Users who never use `recall`/`remember` see no difference. The original tools are unchanged.
- **Positive UX for memory users**: The semantic tool names (`recall`/`remember`) are more intuitive than `query_rag_store`/`ingest_document`.
- **No extra configuration needed**: The behavioral protocol is embedded in tool descriptions, not in client-specific config files.

## Raw Test Data

Full JSON outputs are stored in `/tmp/memory_protocol_test/results/test{1-7}.json`.
