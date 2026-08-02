# GraphRAG — Literature Review & Retrieval Design Notes

Research notes and design specs on graph-augmented retrieval, working from the GraphRAG literature toward a concrete architecture for a large-scale job marketplace search stack.

The through-line: **most of what people want from "GraphRAG" in an industrial retrieval system does not come from porting GraphRAG.** It comes from precomputing structure you already have onto a small, stable entity layer, and injecting it at the right point in an existing dense-retrieval pipeline.

---

## Contents

### 1. [Literature review](graphrag-literature-review.md)
Survey of the GraphRAG field, weighted toward industrial deployment rather than benchmark results. Covers the four phases of the field (proof of concept → cost panic → measurement/skepticism → productionization), the cost-reduction thread (LazyGraphRAG, LightRAG, LinearRAG, KET-RAG, Youtu-GraphRAG), documented production deployments (LinkedIn, Ant Group KAG, ByteDance, Tencent), the benchmark evidence (GraphRAG-Bench, WildGraphBench, BenchmarkQED), and open problems.

**Takeaway:** three independent teams converged on the finding that LLM-based relation extraction at index time is largely optional.

### 2. [Job marketplace graph retrieval design](job-marketplace-graph-retrieval-design.md)
Why porting Microsoft GraphRAG to a marketplace retrieval stack is the wrong move, what the graph actually is (it's already in your databases), four injection points ranked by ROI, and the two-layer scalability argument.

**Takeaway:** route all graph reasoning through a small stable entity layer; precompute job-level neighbor lists. Cost becomes independent of item count.

### 3. [Graph-derived query context](graph-derived-query-context.md)
The `query + context` framing done properly. Query drift in single-vector bi-encoders, why the governing literature is dense query expansion / PRF rather than GraphRAG, verbalization formats, five fusion architectures, and the doc-side-expansion counter-argument.

**Takeaway:** graph context is a vocabulary translation layer between user register and document register.

### 4. [Query enrichment spec](query-enrichment-spec.md)
Implementation-level spec: slot filling rather than graph search, slot inventory, within-slot ranking with a document-frequency band, verbalization templates, α routing, caching, worked example, evaluation ladder.

### 5. [When multi-turn vs. single-turn](when-multi-turn-vs-single-turn.md)
The precomputation test, the five conditions that genuinely force multi-turn, a four-tier architecture (single-turn / stateful refinement / structured query generation / agentic planning), and routing by observed failure rather than predicted intent.

**Takeaway:** multi-hop retrieval ≠ multi-turn agent. A graph collapses multi-hop into single-hop at index time.

### 6. [Multi-turn agent design](multi-turn-agent-design.md)
Tier 3/4 as a typed state machine rather than a chat: bounded action space, termination guarantees, the constraint lattice, verification with evidence spans, prompt-injection surface in user-generated document text.

### 7. [Tier 4 planning agent](tier4-planning-agent.md)
The goal-planning case that genuinely cannot be precomputed. Out-of-band architecture that emits durable state to condition fast retrieval, Pareto path selection, grounding in live inventory, and the survivorship-bias correction for transition graphs.

---

## Suggested reading order

Start at (1) for the field, (2) for the architecture argument, then (3)–(4) for the retrieval-side design or (5)–(7) for the interaction-side design.

## Status

Working notes. Design specs are proposals, not implemented systems. Claims about published work are cited inline in the literature review.
