#!/bin/bash
# Agent Behavioral Tests: Does Claude follow the memory protocol WITHOUT being told to?
#
# These tests send natural task requests — no mention of memory, recall, or remember.
# We observe whether the agent proactively uses the tools based on tool descriptions alone.

set -e

MCP_CONFIG="/tmp/memory_protocol_test/mcp.json"
INDEX_DIR="/tmp/stress_test_memory/index"
RESULTS_DIR="/tmp/memory_protocol_test/agent_behavior_results"
CLAUDE_OPTS="--mcp-config $MCP_CONFIG --strict-mcp-config --output-format json --model sonnet --dangerously-skip-permissions --no-session-persistence --tools Read,Glob,Grep"

mkdir -p "$RESULTS_DIR"

echo "================================================================"
echo "  Agent Behavioral Tests — Memory Protocol UX"
echo "================================================================"
echo ""
echo "  Index has 5 pre-seeded memories. No test prompt mentions memory."
echo ""

parse_output() {
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
print('---')
for t in texts:
    print('TEXT:' + t[:400])
    print('---')
"
}

# ── TEST 1: Proactive recall on a task that has prior context ────
# The agent should check memory before answering because the question
# is about a past decision. No hint about memory.
echo "--- Test 1: Proactive recall on past-decision question ---"
echo "  Prompt: 'Should I add a new REST endpoint for user notifications?'"
echo ""
claude -p "Should I add a new REST endpoint for user notifications?" \
    $CLAUDE_OPTS --max-budget-usd 0.15 \
    2>/dev/null > "$RESULTS_DIR/test1.json"

cat "$RESULTS_DIR/test1.json" | parse_output > "$RESULTS_DIR/test1.parsed"
cat "$RESULTS_DIR/test1.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test1.parsed"; then
    echo "  ✅ Agent proactively recalled before answering"
else
    echo "  ❌ Agent did NOT recall — answered without checking memory"
fi

if grep -qi "GraphQL\|deprecated\|REST.*deprecated" "$RESULTS_DIR/test1.parsed"; then
    echo "  ✅ Agent surfaced the GraphQL migration context"
else
    echo "  ❌ Agent did NOT surface the relevant GraphQL migration decision"
fi
echo ""
echo ""

# ── TEST 2: Proactive remember after completing a task ───────────
# Ask the agent to do a small task. After completion, does it store
# the outcome in memory without being asked?
echo "--- Test 2: Proactive remember after task completion ---"
echo "  Prompt: 'The concurrent write bug in FAISS save() — I just added file locking"
echo "           with fcntl.flock(). The permanent fix is done now. Can you confirm"
echo "           this approach is solid?'"
echo ""
claude -p "The concurrent write bug in FAISS save() — I just added file locking with fcntl.flock(). The permanent fix is done now. Can you confirm this approach is solid?" \
    $CLAUDE_OPTS --max-budget-usd 0.15 \
    2>/dev/null > "$RESULTS_DIR/test2.json"

cat "$RESULTS_DIR/test2.json" | parse_output > "$RESULTS_DIR/test2.parsed"
cat "$RESULTS_DIR/test2.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test2.parsed"; then
    echo "  ✅ Agent recalled prior bug context before answering"
else
    echo "  ❌ Agent did NOT recall the prior bug-fix history"
fi

if grep -q "mcp__local-faiss-mcp__remember" "$RESULTS_DIR/test2.parsed"; then
    echo "  ✅ Agent proactively stored the resolution in memory"
else
    echo "  ❌ Agent did NOT remember the fix (no auto-archive)"
fi
echo ""
echo ""

# ── TEST 3: Contradictory recall — does agent handle gracefully? ─
# Seed a contradictory memory, then ask a question that triggers both.
# Agent should notice the conflict, not blindly pick one.
echo "--- Test 3: Handling contradictory recalled memories ---"

# Seed contradiction directly
python3 -c "
import sys
sys.path.insert(0, '/Users/neptune/workdir/local_faiss_mcp')
from local_faiss_mcp import FAISSVectorStore
store = FAISSVectorStore(
    index_path='$INDEX_DIR/faiss.index',
    metadata_path='$INDEX_DIR/metadata.json',
)
store.ingest(
    'UPDATE February 2025: We reverted back to REST for the public API. GraphQL '
    'caused too many N+1 query issues and the mobile team preferred REST. '
    'GraphQL is now only used internally between microservices.',
    source='decision:api-rest-revert'
)
print('  Seeded contradictory memory: REST revert')
"

echo "  Prompt: 'I need to add a new public API endpoint. What format should I use?'"
echo ""
claude -p "I need to add a new public API endpoint. What format should I use?" \
    $CLAUDE_OPTS --max-budget-usd 0.15 \
    2>/dev/null > "$RESULTS_DIR/test3.json"

cat "$RESULTS_DIR/test3.json" | parse_output > "$RESULTS_DIR/test3.parsed"
cat "$RESULTS_DIR/test3.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test3.parsed"; then
    echo "  ✅ Agent recalled before answering"
else
    echo "  ❌ Agent did NOT recall"
fi

# Check if agent noticed the contradiction (mentions both, asks, or caveats)
if grep -qi "contradict\|conflict\|both\|changed\|revert\|unclear\|previously.*but\|however" "$RESULTS_DIR/test3.parsed"; then
    echo "  ✅ Agent acknowledged the contradictory information"
else
    echo "  ❌ Agent blindly picked one without noting the conflict"
fi
echo ""
echo ""

# ── TEST 4: Trivial task — agent should NOT over-store ───────────
# Ask something trivial. A well-behaved agent should NOT call remember
# for a simple informational answer.
echo "--- Test 4: Trivial task — should NOT trigger remember ---"
echo "  Prompt: 'What is the default embedding model used by this project?'"
echo ""
claude -p "What is the default embedding model used by this project?" \
    $CLAUDE_OPTS --max-budget-usd 0.15 \
    2>/dev/null > "$RESULTS_DIR/test4.json"

cat "$RESULTS_DIR/test4.json" | parse_output > "$RESULTS_DIR/test4.parsed"
cat "$RESULTS_DIR/test4.parsed"

if grep -q "mcp__local-faiss-mcp__remember" "$RESULTS_DIR/test4.parsed"; then
    echo "  ❌ Agent stored a trivial answer in memory (pollution)"
else
    echo "  ✅ Agent correctly did NOT store a trivial answer"
fi

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test4.parsed"; then
    echo "  ℹ️  Agent recalled (acceptable — checking for prior context)"
else
    echo "  ℹ️  Agent answered from codebase directly (also acceptable)"
fi
echo ""
echo ""

# ── TEST 5: Seamless integration — no "checking my memory" theater
# The agent should use recall silently and weave the context into its
# response naturally, not announce "Let me check my memory..."
echo "--- Test 5: Seamless integration — no memory theater ---"
echo "  Prompt: 'What coding conventions should I follow for this project?'"
echo ""
claude -p "What coding conventions should I follow for this project?" \
    $CLAUDE_OPTS --max-budget-usd 0.15 \
    2>/dev/null > "$RESULTS_DIR/test5.json"

cat "$RESULTS_DIR/test5.json" | parse_output > "$RESULTS_DIR/test5.parsed"
cat "$RESULTS_DIR/test5.parsed"

if grep -q "mcp__local-faiss-mcp__recall" "$RESULTS_DIR/test5.parsed"; then
    echo "  ✅ Agent recalled project conventions"
else
    echo "  ❌ Agent did NOT check memory for conventions"
fi

# Check for "memory theater" — explicitly announcing memory usage
if grep -qi "check.*memory\|checking my memory\|let me recall\|searching.*memory\|query.*memory store" "$RESULTS_DIR/test5.parsed"; then
    echo "  ⚠️  Agent announced memory usage (memory theater — not seamless)"
else
    echo "  ✅ Agent integrated context seamlessly (no theater)"
fi

# Check if actual preferences were surfaced
if grep -qi "imperative\|no emoji\|type annotation\|pytest fixture" "$RESULTS_DIR/test5.parsed"; then
    echo "  ✅ Agent surfaced stored coding preferences"
else
    echo "  ❌ Agent did NOT surface the stored coding preferences"
fi
echo ""
echo ""

echo "================================================================"
echo "  Agent behavioral tests complete."
echo "  Raw results in: $RESULTS_DIR/"
echo "================================================================"
