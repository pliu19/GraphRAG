# GraphRAG at Industrial Scale — Literature Review

*Compiled 2026-08-01. Emphasis on production/industrial deployment rather than academic benchmark chasing.*

---

## 1. Framing: the one-sentence version

GraphRAG replaces "retrieve top-k chunks by embedding similarity" with "build an entity–relation graph over the corpus, optionally summarize its dense regions, and retrieve over that structure." The claimed payoff is multi-hop reasoning and corpus-level *sensemaking* ("what are the main themes?") that flat vector RAG structurally cannot do.

The last two years of literature can be read as a single arc:

| Phase | Period | Question being asked |
|---|---|---|
| I. Proof of concept | 2024 | Can graphs beat vector RAG at all? |
| II. Cost panic | late 2024 – 2025 | It works but indexing costs $10k–$33k per corpus. Can we make it cheap? |
| III. Skepticism + measurement | 2025 – 2026 | *Does* it actually beat vector RAG, on what, and how do we tell? |
| IV. Productionization | 2025 – 2026 | Incremental updates, latency SLAs, schema governance, agent memory |

If you're writing a survey or a system, Phase II and III are where the interesting industrial content lives. Phase I is well-covered ground.

---

## 2. Foundational papers (read these first)

**Edge et al., "From Local to Global: A Graph RAG Approach to Query-Focused Summarization" (Microsoft, arXiv 2404.16130)** — the paper that started the wave. Key structural idea: LLM-extracted entity graph → Leiden community detection → hierarchical LLM-generated community summaries → map-reduce over summaries for *global* queries. Reported substantial gains in **comprehensiveness and diversity** over vector RAG on ~1M-token corpora. Note the honest scoping: they claim wins on *query-focused summarization*, not on factoid QA. Much of the later "GraphRAG doesn't work" discourse comes from people evaluating it on tasks it never claimed.

**Peng et al., "Graph Retrieval-Augmented Generation: A Survey" (arXiv 2408.08921)** — the canonical taxonomy: *graph-based indexing → graph-guided retrieval → graph-enhanced generation*. Use this decomposition as the spine of any review; nearly every subsequent paper is an intervention at exactly one of these three stages.

**Han et al., "Retrieval-Augmented Generation with Graphs (GraphRAG)" (arXiv 2501.00309)** and **"A Survey of GraphRAG for Customized LLMs" (arXiv 2501.13958)** — broader, domain-organized surveys. The DEEP-PolyU [Awesome-GraphRAG](https://github.com/DEEP-PolyU/Awesome-GraphRAG) list is the best-maintained tracker and organizes work as *knowledge organization / knowledge retrieval / knowledge integration*.

**Method families worth knowing by name:**

- **Community-summary** (Microsoft GraphRAG, ArchRAG) — global sensemaking, expensive indexing.
- **Dual-level / lightweight** (LightRAG, EMNLP 2025) — local + global keyword retrieval over a KG plus vector index; explicit **incremental update algorithm**, which is the feature that matters industrially.
- **Text-centric, graph-as-router** (HippoRAG, HippoRAG 2) — graph indexes *passages*; Personalized PageRank gives one-shot multi-hop retrieval. Cheapest conceptual bridge from existing vector stacks.
- **Hierarchical summary, no explicit graph** (RAPTOR) — useful ablation: how much of GraphRAG's benefit is the *graph* vs. just *hierarchical summarization*?
- **Schema-guided / agentic** (KAG, Youtu-GraphRAG, Think-on-Graph 1/2/3) — a domain schema constrains extraction; agents traverse at query time.
- **Graph-free / adaptive** (LazyGraphRAG, StructRAG, "You Don't Need Pre-built Graphs for RAG", AAAI 2026) — defer or skip graph construction entirely.

---

## 3. The industrial-scale problem, stated precisely

Four distinct failure modes block production deployment. Most papers address exactly one; conflating them is the most common weakness in reviews of this area.

**(a) Indexing cost.** Canonical GraphRAG makes an LLM call per chunk for extraction, then more per community for summarization. Practitioner reports of **~$33k to index a large corpus** are widely cited as the moment enterprise interest stalled. This is the single most-attacked problem in the literature.

**(b) Incremental update / staleness.** Enterprise corpora change hourly. Naive community detection + summarization is a *global batch* operation — one new document can in principle perturb the community structure. Any system that requires full reindexing is dead on arrival for ticketing systems, code repos, or news.

**(c) Query-time cost and latency.** Global search in the original design map-reduces over *all* community summaries. LazyGraphRAG reports **>700× lower query cost** than GraphRAG global search at comparable quality — which tells you how bad the baseline was. Production QA needs sub-second-to-few-second p95.

**(d) Extraction quality ceiling.** The most underrated finding in the whole literature: in the systematic RAG-vs-GraphRAG study, **only ~65.8% of answer entities appeared in the constructed KG for HotpotQA**. Retrieval quality is capped by extraction recall, and no amount of clever traversal fixes a missing node. Schema-guided extraction (KAG, Youtu-GraphRAG) is the main response.

---

## 4. The cost-reduction literature (Phase II — the densest thread)

This is where I'd focus if the review is meant to be industrially useful.

| Work | Lever pulled | Headline claim |
|---|---|---|
| **LazyGraphRAG** (Microsoft, 2024) | Defer *all* LLM summarization to query time; index with cheap NLP only | Indexing cost = vector RAG's, i.e. **0.1% of full GraphRAG**; **>700×** lower query cost at comparable global-query quality |
| **LightRAG** (HKUDS, EMNLP 2025) | Dual-level keyword retrieval; incremental graph union | Retrieval in **<100 tokens** where GraphRAG needs ~610k; **~50%** faster incremental updates |
| **KET-RAG** (2025) | Multi-granular indexing — expensive graph only on a skeleton of important chunks, cheap keyword-bipartite for the rest | Cost-efficient indexing at retained quality |
| **E²GraphRAG** (2025) | Streamlined construction + retrieval | Efficiency/effectiveness Pareto |
| **LinearRAG** (ICLR 2026) | **Relation-free** graph construction — bipartite entity–passage graph, no LLM relation extraction | Linear-cost indexing on large-scale corpora |
| **Youtu-GraphRAG** (Tencent, ICLR 2026) | Schema-guided extraction + dual-perception communities + agentic retrieval, vertically unified | **90.71%** token-cost reduction and **+16.62%** accuracy vs. SOTA baselines; **33.6%** lower token cost at the Pareto frontier |
| **Towards Practical GraphRAG** (arXiv 2507.03226) | Replace LLM extraction with **dependency parsing**; hybrid vector+traversal fused by RRF | **94%** of LLM-extraction quality (61.87 vs 65.83) at a fraction of cost; +15% / +4.35% over vector baselines on enterprise legacy-code data |
| **Core-based Hierarchies for Efficient GraphRAG** (KDD 2026) | Replace Leiden communities with $k$-core–based hierarchy | Structural indexing alternative — worth reading for the indexing-primitive argument |

**The synthesis worth writing down:** three independent teams — LazyGraphRAG, LinearRAG, and *Towards Practical GraphRAG* — converged on the same conclusion from different directions: **LLM-based relation extraction at index time is largely optional.** You can get most of the benefit from cheap structure (co-occurrence, dependency parsing, entity–passage bipartite graphs) plus intelligence deferred to query time. That is the single most important industrial takeaway of 2025–26. Any new system that burns an LLM call per chunk now bears the burden of proof.

---

## 5. Documented industrial deployments

Genuine production evidence with numbers is scarce; here's what exists.

**LinkedIn — customer service QA (SIGIR 2024, arXiv 2404.17723).** The best-documented industrial GraphRAG deployment. Builds a KG from historical issue tickets preserving *intra-issue tree structure* and *inter-issue relations*, rather than flattening tickets to text. Reported **+77.6% MRR** and **+0.32 BLEU** over baseline, and after ~6 months in production, a **28.6% reduction in median per-issue resolution time**. ⚠️ Secondary sources circulate a "40 hours → 15 hours, 63%" figure; the paper's own claim is the 28.6% median figure — cite the paper.

**Ant Group — KAG / OpenSPG (arXiv 2409.13731).** Knowledge-Augmented Generation: mutual indexing between a domain-schema-constrained KG and text chunks, logical-form-guided hybrid reasoning, explicitly designed to fix the "vector similarity is not logical relevance" problem. Deployed in Alipay's *ZhiXiaoBao*: reported **91% accuracy on government-service inquiries** and **>90% on medical metric interpretation**. Open-sourced via [OpenSPG](https://github.com/OpenSPG/openspg), built on DB-GPT + TuGraph. This is the most complete *stack* story — schema, engine, graph DB, application — and the closest thing to a reference architecture.

**ByteDance.** Per the LLM+Graph@VLDB'2025 workshop summary (arXiv 2604.02861), GraphRAG is **deployed across 50+ scenarios**, with a stated long-term goal of a unified **Graph Foundation Model** trained once and applied across business domains. Backed by their own graph infrastructure (ByteGraph, BG3). Little public methodological detail — cite as adoption evidence, not as method.

**Tencent Cloud.** Youtu-GraphRAG shipped as an **Enterprise Edition on the Tencent Cloud ADP platform**, with schema-driven domain transfer as the stated adaptation mechanism.

**Microsoft.** GraphRAG as an OSS library plus LazyGraphRAG, plus **BenchmarkQED** (AutoQ / AutoE / AutoD) for automated benchmark generation and evaluation, plus integration into Azure (Microsoft Discovery). Effectively defines the reference implementation and the reference evaluation harness.

**Vendor/infrastructure layer.** Neo4j (+ Graphiti for agent memory), Amazon Neptune Analytics + Bedrock, FalkorDB GraphRAG SDK, TuGraph/Chat2Graph, TigerVector (vector search inside a graph DB, arXiv 2501.11216). Gartner has GraphRAG in its 2026 top D&A trends. Treat vendor material as directional, not evidentiary.

**Domain verticals with published case studies:** healthcare (KG-RAG on PubMedQA, UMLS/SNOMED-grounded clinical decision support), industrial safety (LOTO procedure failure analysis), manufacturing document QA, supply chain configuration, financial cross-entity sentiment. These are mostly single-site studies — useful as existence proofs, weak as evidence.

---

## 6. Does it actually work? (Phase III — read this before believing anything above)

**"RAG vs. GraphRAG: A Systematic Evaluation and Key Insights" (arXiv 2502.11371).** The most useful single paper for a skeptical practitioner. Compares vector RAG against KG-based (LlamaIndex), community-based (Microsoft), text-centric (HippoRAG2), and hierarchical-summary (RAPTOR) variants, on NQ / HotpotQA / MultiHop-RAG / NovelQA and four summarization sets.

Findings:
- GraphRAG wins on **multi-hop, comparison, temporal, and global/corpus-level** queries.
- Vector RAG wins by **11–14%** on **single-hop, detail-oriented factoid** queries.
- Global retrieval **increases hallucination risk on null queries** (no answer in corpus) — a real production hazard.
- Extraction recall is the binding constraint (**~65.8%** answer-entity coverage).
- **Hybrid strategies both work**: *Selection* (route by query type) and *Integration* (run both, concatenate). Integration gives up to **+6.4%** on multi-hop but doubles compute.

**GraphRAG-Bench (ICLR 2026).** Four-tier task ladder — fact retrieval → complex reasoning → contextual summarization → creative generation — built explicitly to answer *when* graphs help. Its framing concedes the field's uncomfortable fact: GraphRAG frequently underperforms vanilla RAG in practice, so the research question is now conditional, not absolute.

**WildGraphBench (arXiv 2602.02053).** The most industrially relevant benchmark: evaluates Local-Global GraphRAG, LightRAG, FastGraphRAG, LinearRAG, RAG-Memory and others on **"wild-source" corpora** rather than curated academic sets. Finding: **significant performance degradation on wild corpora vs. curated datasets**, with construction/retrieval cost the practical blocker. If your review has one "what the field is missing" claim, this is the evidence for it.

**BenchmarkQED (Microsoft).** AutoQ synthesizes queries across a local↔global spectrum, AutoE does LLM-judged pairwise comparison on comprehensiveness/diversity/empowerment, AutoD handles data prep. Released datasets: AP News (1,397 health articles) and Behind the Tech podcast transcripts. Adopt this if you need to run your own evaluation — the alternative is inventing metrics nobody trusts.

**Honest bottom line:** the *sign* of GraphRAG's effect depends on query type; the *magnitude* depends on extraction quality; and the *viability* depends on cost. Papers claiming unconditional superiority are almost always evaluating on curated multi-hop QA with a favorable corpus.

---

## 7. Adjacent thread: temporal graphs and agent memory

Worth including because industrially it may end up bigger than document QA.

**Zep / Graphiti (arXiv 2501.13956).** Temporally-aware KG engine for agent memory: incrementally ingests conversational *and* structured business data, updates entities/relations/communities **without batch recomputation**, and maintains **validity intervals** for facts (non-lossy invalidation rather than overwrite). Reports **94.8% vs 93.4%** over MemGPT on DMR, up to **+18.5%** accuracy on LongMemEval with **~90% lower latency**.

This solves problem (b) from §3 — incremental update — more convincingly than most document-QA GraphRAG systems, because it was designed for a domain where staleness is the whole problem. The bi-temporal modeling (event time vs. ingestion time) is a design pattern the document-QA branch has largely not adopted.

---

## 8. Open problems / where the gaps are

1. **Extraction recall is the bottleneck, and almost nobody measures it directly.** Most papers report end-to-end QA accuracy. The 65.8% coverage number suggests intrinsic KG-construction evaluation is badly underdeveloped.
2. **No standard cost accounting.** Papers report tokens, dollars, wall-clock, or nothing, on different corpora and models. Cross-paper cost claims are essentially incomparable — a genuine contribution opportunity.
3. **Incremental maintenance for community-based methods.** LightRAG unions new nodes; but what happens to *community summaries* when the community structure drifts? Largely unanswered.
4. **Query routing is under-studied relative to its value.** Selection-based hybrids get most of GraphRAG's benefit at a fraction of the cost, yet the routing classifier itself is treated as an afterthought.
5. **Wild-corpus robustness.** WildGraphBench shows the academic→real gap is large. Noisy OCR, duplicates, contradictions, multilingual and multi-tenant corpora are barely addressed.
6. **Multi-tenancy, ACLs, and governance.** Essentially absent from the academic literature and a hard blocker in enterprises: graph traversal can leak across permission boundaries in ways chunk-level ACL filtering does not.
7. **Serving systems.** The systems community (VLDB workshop) frames this as reasoning over billion-scale property graphs, consistency of LLM-written graph updates, and LLM/DB co-design — a mostly separate conversation from the NLP-side GraphRAG literature. Bridging them is open.
8. **Graph Foundation Models.** ByteDance's stated direction: train once, apply across domains. No convincing public results yet.

---

## 9. Suggested reading path

**If you have 3 papers:** Edge et al. 2404.16130 (what it is) → arXiv 2502.11371 (whether it works) → LazyGraphRAG blog (how to afford it).

**If you have 8:** add Peng et al. 2408.08921 (taxonomy), LinkedIn 2404.17723 (production proof), KAG 2409.13731 (schema-driven industrial stack), LightRAG (incremental updates), WildGraphBench 2602.02053 (reality check).

**If you're building:** LinearRAG + Youtu-GraphRAG + *Towards Practical GraphRAG* for the cheap-construction argument; BenchmarkQED to evaluate; Graphiti if state is conversational rather than documentary.

---

## Sources

- [From Local to Global: A Graph RAG Approach to Query-Focused Summarization (arXiv 2404.16130)](https://arxiv.org/abs/2404.16130)
- [Graph Retrieval-Augmented Generation: A Survey (arXiv 2408.08921)](https://arxiv.org/abs/2408.08921)
- [A Survey of GraphRAG for Customized LLMs (arXiv 2501.13958)](https://arxiv.org/abs/2501.13958)
- [Awesome-GraphRAG (DEEP-PolyU)](https://github.com/DEEP-PolyU/Awesome-GraphRAG)
- [LazyGraphRAG sets a new standard for GraphRAG quality and cost (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/lazygraphrag-setting-a-new-standard-for-quality-and-cost/)
- [Project GraphRAG (Microsoft Research)](https://www.microsoft.com/en-us/research/project/graphrag/)
- [BenchmarkQED: Automated benchmarking of RAG systems (Microsoft Research)](https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/) · [repo](https://github.com/microsoft/benchmark-qed)
- [LightRAG: Simple and Fast Retrieval-Augmented Generation (EMNLP 2025)](https://github.com/hkuds/lightrag)
- [Youtu-GraphRAG (ICLR 2026, arXiv 2508.19855)](https://arxiv.org/pdf/2508.19855) · [Tencent repo](https://github.com/TencentCloudADP/youtu-graphrag)
- [Towards Practical GraphRAG: Efficient KG Construction and Hybrid Retrieval at Scale (arXiv 2507.03226)](https://arxiv.org/abs/2507.03226)
- [Core-based Hierarchies for Efficient GraphRAG (KDD 2026, arXiv 2603.05207)](https://arxiv.org/pdf/2603.05207)
- [RAG vs. GraphRAG: A Systematic Evaluation and Key Insights (arXiv 2502.11371)](https://arxiv.org/html/2502.11371v3)
- [GraphRAG-Bench (ICLR 2026)](https://graphrag-bench.github.io/) · [repo](https://github.com/GraphRAG-Bench/GraphRAG-Benchmark)
- [WildGraphBench: Benchmarking GraphRAG with Wild-Source Corpora (arXiv 2602.02053)](https://arxiv.org/pdf/2602.02053)
- [Retrieval-Augmented Generation with Knowledge Graphs for Customer Service QA — LinkedIn (SIGIR 2024, arXiv 2404.17723)](https://arxiv.org/abs/2404.17723)
- [KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation (arXiv 2409.13731)](https://arxiv.org/abs/2409.13731) · [OpenSPG](https://github.com/OpenSPG/openspg)
- [LLM+Graph@VLDB'2025 Workshop Summary (arXiv 2604.02861)](https://arxiv.org/pdf/2604.02861)
- [Zep: A Temporal Knowledge Graph Architecture for Agent Memory (arXiv 2501.13956)](https://arxiv.org/abs/2501.13956) · [Graphiti / Neo4j](https://neo4j.com/blog/developer/graphiti-knowledge-graph-memory/)
- [ArchRAG: Attributed Community-based Hierarchical RAG (arXiv 2502.09891)](https://arxiv.org/pdf/2502.09891)
- [LEGO-GraphRAG: Modularizing Graph-based RAG for Design Space Exploration (arXiv 2411.05844)](https://arxiv.org/pdf/2411.05844)
- [TigerVector: Supporting Vector Search in Graph Databases (arXiv 2501.11216)](https://arxiv.org/pdf/2501.11216)
- [You Don't Need Pre-built Graphs for RAG (AAAI 2026, arXiv 2508.06105)](https://arxiv.org/html/2508.06105v2)
- [Think-on-Graph 3.0 (arXiv 2509.21710)](https://arxiv.org/pdf/2509.21710)
- [Evaluating GraphRAG for industrial safety: LOTO procedure failures (ScienceDirect)](https://www.sciencedirect.com/science/article/pii/S1877050926003637)
