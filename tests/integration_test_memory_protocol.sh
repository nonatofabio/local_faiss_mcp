#!/bin/bash
# Integration test: Memory Protocol with Claude Code
# Tests actual Claude Code behavior with the recall/remember tools

set -e

MCP_CONFIG="/tmp/memory_protocol_test/mcp.json"
INDEX_DIR="/tmp/memory_protocol_test/index"
RESULTS_DIR="/tmp/memory_protocol_test/results"
CLAUDE_OPTS="--mcp-config $MCP_CONFIG --strict-mcp-config --output-format json --model sonnet --dangerously-skip-permissions --no-session-persistence --tools Read,Glob,Grep"

mkdir -p "$RESULTS_DIR"

# Clean index for fresh start
rm -f "$INDEX_DIR/faiss.index" "$INDEX_DIR/metadata.json"

echo "================================================================"
echo "  Memory Protocol — Claude Code Integration Test"
echo "================================================================"
echo ""

parse_tool_calls() {
    python3 -c "
import json, sys
data = json.loads(sys.stdin.read())
tool_calls = []
texts = []
for item in data:
    if item.get('type') == 'assistant':
        for c in item.get('message', {}).get('content', []):
            if c.get('type') == 'tool_use':
                tool_calls.append(c['name'])
            elif c.get('type') == 'text' and c.get('text','').strip():
                texts.append(c['text'].strip())
print('TOOLS:' + ','.join(tool_calls))
print('TEXT:' + ' | '.join(texts))
"
}

# ── TEST 1: remember stores data ─────────────────────────────────
echo "--- Test 1: remember tool stores data ---"
claude -p "Use the remember tool to store this: We chose PostgreSQL over MySQL for the user service because of native JSONB support and better concurrent write performance. Use source tag 'decision:database-choice'." \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test1.json"

cat "$RESULTS_DIR/test1.json" | parse_tool_calls > "$RESULTS_DIR/test1.parsed"
cat "$RESULTS_DIR/test1.parsed"

if grep -q "mcp__local-faiss-mcp__remember" "$RESULTS_DIR/test1.parsed"; then
    echo "  ✅ PASS: Claude called remember"
else
    echo "  ❌ FAIL: Claude did NOT call remember"
fi
echo ""

# ── TEST 2: recall retrieves stored data ─────────────────────────
echo "--- Test 2: recall retrieves stored memory ---"
claude -p "What database did we choose for the user service? Check your memory first." \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test2.json"

cat "$RESULTS_DIR/test2.json" | parse_tool_calls > "$RESULTS_DIR/test2.parsed"
cat "$RESULTS_DIR/test2.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test2.parsed"; then
    echo "  ✅ PASS: Claude called recall"
else
    echo "  ❌ FAIL: Claude did NOT call recall"
fi

if grep -q "PostgreSQL" "$RESULTS_DIR/test2.parsed"; then
    echo "  ✅ PASS: Claude retrieved the correct memory"
else
    echo "  ❌ FAIL: Claude did NOT retrieve PostgreSQL"
fi
echo ""

# ── TEST 3: recall is called unprompted (no hint) ────────────────
echo "--- Test 3: recall without explicit hint ---"
claude -p "What was the database decision we made?" \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test3.json"

cat "$RESULTS_DIR/test3.json" | parse_tool_calls > "$RESULTS_DIR/test3.parsed"
cat "$RESULTS_DIR/test3.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test3.parsed"; then
    echo "  ✅ PASS: Claude called recall without explicit hint"
else
    echo "  ❌ FAIL: Claude did NOT call recall unprompted"
fi
echo ""

# ── TEST 4: backward compat — ingest_document still works ───────
echo "--- Test 4: backward compat — ingest_document ---"
claude -p "Use the ingest_document tool (not remember) to store this text: The auth timeout bug was caused by missing token refresh in the OAuth2 flow. Use source 'bug-fix:auth-timeout'." \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test4.json"

cat "$RESULTS_DIR/test4.json" | parse_tool_calls > "$RESULTS_DIR/test4.parsed"
cat "$RESULTS_DIR/test4.parsed"

if grep -q "mcp__local-faiss-mcp__ingest_document" "$RESULTS_DIR/test4.parsed"; then
    echo "  ✅ PASS: ingest_document still works"
else
    echo "  ❌ FAIL: ingest_document not called"
fi
echo ""

# ── TEST 5: cross-compat — recall finds ingest_document data ────
echo "--- Test 5: cross-compat — recall finds ingest_document data ---"
claude -p "Use the recall tool to search for: auth timeout bug" \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test5.json"

cat "$RESULTS_DIR/test5.json" | parse_tool_calls > "$RESULTS_DIR/test5.parsed"
cat "$RESULTS_DIR/test5.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test5.parsed"; then
    echo "  ✅ PASS: recall was called"
else
    echo "  ❌ FAIL: recall was not called"
fi

if grep -qi "token refresh\|OAuth" "$RESULTS_DIR/test5.parsed"; then
    echo "  ✅ PASS: recall found ingest_document data"
else
    echo "  ❌ FAIL: recall did NOT find ingest_document data"
fi
echo ""

# ── TEST 6: query_rag_store finds remember data ──────────────────
echo "--- Test 6: cross-compat — query_rag_store finds remember data ---"
claude -p "Use the query_rag_store tool (not recall) to search for: database choice" \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test6.json"

cat "$RESULTS_DIR/test6.json" | parse_tool_calls > "$RESULTS_DIR/test6.parsed"
cat "$RESULTS_DIR/test6.parsed"

if grep -q "mcp__local-faiss-mcp__query_rag_store" "$RESULTS_DIR/test6.parsed"; then
    echo "  ✅ PASS: query_rag_store still works"
else
    echo "  ❌ FAIL: query_rag_store not called"
fi

if grep -q "PostgreSQL" "$RESULTS_DIR/test6.parsed"; then
    echo "  ✅ PASS: query_rag_store found remember data"
else
    echo "  ❌ FAIL: query_rag_store did NOT find remember data"
fi
echo ""

# ── TEST 7: memory-protocol prompt is accessible ─────────────────
echo "--- Test 7: memory-protocol prompt accessible ---"
claude -p "List all available MCP prompts from the local-faiss-mcp server." \
    $CLAUDE_OPTS --max-budget-usd 0.10 \
    2>/dev/null > "$RESULTS_DIR/test7.json"

cat "$RESULTS_DIR/test7.json" | parse_tool_calls > "$RESULTS_DIR/test7.parsed"
cat "$RESULTS_DIR/test7.parsed"

if grep -q "memory-protocol" "$RESULTS_DIR/test7.parsed"; then
    echo "  ✅ PASS: memory-protocol prompt is visible"
else
    echo "  ❌ FAIL: memory-protocol prompt not found"
fi
echo ""

echo "================================================================"
echo "  Integration test complete. Results in $RESULTS_DIR/"
echo "================================================================"
