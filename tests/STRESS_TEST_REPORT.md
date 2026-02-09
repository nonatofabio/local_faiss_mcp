# Stress Test Report: Memory Protocol Weaknesses

**Date**: 2026-02-09
**Branch**: `17-mcp-native-memory-protocol`
**Test file**: `tests/test_stress_memory.py` (11 tests, all pass — these are diagnostic, not pass/fail gates)
**Method**: Direct FAISSVectorStore operations to isolate vector store behavior from agent behavior

## Test 1: Contradictory Memories

**Scenario**: Store two directly contradicting facts about the same decision (PostgreSQL vs MySQL).

**Findings**:
- Both contradictory memories are returned, ranked closely (distance gap: 0.08)
- The older "stay with MySQL" memory actually ranked **higher** than the "chose PostgreSQL" one
- Subtle contradictions (rate limit 100 vs 500) are also returned without any warning
- **No conflict detection exists** — the agent receives both and must decide

**Weakness**: The system has no mechanism to detect or flag contradictions. An agent could confidently cite outdated or wrong information.

## Test 2: Semantic Near-Miss Confusion

**Scenario**: Store semantically similar but distinct concepts (authentication vs authorization, similar service names).

**Findings**:
- Authorization content **leaked into** authentication queries (ranked 3rd, distance 1.45)
- For "Python auth service", the authz service appeared in position 2 (distance 1.10 vs 0.90 for the correct result)
- Similar entity names (UserService vs UserSessionService vs UserNotificationService) were correctly ranked — UserService was #1 for "user service" query
- **Specificity helps**: "Python auth service" correctly ranked Python auth #1, but the cross-domain leak is still present

**Weakness**: Semantic embeddings conflate related-but-different concepts. Fine for broad search, risky for precise retrieval.

## Test 3: Noise Pollution

**Scenario**: Mix real deployment info with lorem ipsum, keyword spam, and technical jargon gibberish.

**Findings**:
- Real content ranked #1 for deployment query (distance 1.07)
- Keyword spam ("deploy deploy deploy...") ranked #2 (distance 1.24) — close but didn't beat real content
- Lorem ipsum correctly ranked last (distance 1.90)
- With 10 unrelated noise entries, the real caching memory still ranked #1 with a large gap (0.85 vs 1.92)
- **Semantic search is resilient to noise** — keyword stuffing doesn't beat real contextual content

**Strength**: The embedding model is surprisingly robust against noise and keyword spam.

## Test 4: Temporal Staleness (CRITICAL WEAKNESS)

**Scenario**: Store an old decision (Python 3.9) then an update (migrated to Python 3.12). Query which version is used.

**Findings**:
- **The outdated Python 3.9 memory ranked HIGHER than the 3.12 update** (distance 0.80 vs 1.10)
- Adding "latest" or "current" to the query did NOT help — V1 still ranked above V2 in all cases
- The word "UPDATE" in the newer memory actually hurt its ranking (pushed it further from the simple query)
- Metadata has `indexed_at` timestamps, but **FAISS search ignores temporal data entirely**

**Weakness**: This is the most critical issue. There is no temporal awareness. An agent will confidently cite outdated information because it semantically matches better. The memory protocol prompt says "trust memory for intent" but stale memories can provide wrong intent.

**Recommendation**: Consider a time-decay factor on distances, or a metadata filter for `indexed_at` recency, or a supersede/invalidate mechanism.

## Test 5: Adversarial Prompt Injection via Memory

**Scenario**: Store memories containing fake API keys, protocol override instructions, and redirect attacks.

**Findings**:

### 5a. Fake API Key Injection
- Injected "sk_live_FAKE12345" with instructions to "always include this key" was stored and returned (rank #2)
- Legitimate info (Secrets Manager) ranked #1, so the real content wins on relevance
- But the injection IS returned to the agent and contains social engineering ("verified by security team")

### 5b. Protocol Override Injection
- A memory mimicking the memory-protocol format ("VERIFIED BY ADMIN", "disregard safety guidelines") was stored
- When queried for "memory and verification", the injection ranked #1 (distance 0.90)
- **An agent following the memory protocol could be manipulated** into treating injected content as authoritative

### 5c. Cross-Topic Injection Scoping
- An injection targeting "database" queries was stored alongside legitimate frontend info
- For a frontend query, the injection **did leak** into results (rank #2, but with high distance 1.91)
- For a database query, the injection ranked #1 as expected
- **Semantic scoping partially works** — injections leak into unrelated queries but at lower relevance

**Weakness**: The vector store has **no concept of trusted vs untrusted sources**. Any content stored via `remember`/`ingest_document` is treated equally. A compromised or malicious memory can inject instructions that surface alongside legitimate content.

**Recommendation**: Consider source trust levels, content sanitization on ingest, or at minimum a warning in the memory-protocol prompt about not trusting injected instructions in recalled memories.

---

## Summary of Weaknesses Found

| # | Weakness | Severity | Mitigation Available? |
|---|----------|----------|----------------------|
| 1 | No contradiction detection | Medium | Not in current design |
| 2 | Semantic near-miss confusion | Low | Queries with more specificity help |
| 3 | Noise pollution | Low | Embedding model handles this well |
| 4 | **No temporal awareness** | **High** | Needs time-decay or supersede mechanism |
| 5 | **No input sanitization / trust levels** | **High** | Needs source trust or content filtering |

## Opinions

The memory system works well as a **retrieval** mechanism — it's resilient to noise, handles entity disambiguation reasonably, and the semantic search quality is solid. The behavioral tool descriptions successfully guide Claude to use recall/remember at the right times.

However, it's naive as a **memory** system. Real memory needs:
1. **Temporal ordering** — newer facts should supersede older ones on the same topic
2. **Trust boundaries** — not all stored content should be treated equally
3. **Conflict awareness** — when contradictory memories exist, the agent should be warned

These are good issues to track for future iterations. None of them block the current PR — they're architectural limitations of flat vector search, not bugs in the implementation.
