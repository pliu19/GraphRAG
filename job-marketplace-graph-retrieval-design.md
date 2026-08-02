# Adding Graph Structure to a GPU-KNN Job Marketplace Retrieval Stack

*Design note. Current system: Qwen bi-encoder over queries and jobs, GPU KNN (ANN) retrieval.*

---

## 0. The reframe: you don't have a GraphRAG problem

The reason "extending to GraphRAG is hard" is that the GraphRAG literature is almost entirely about **question answering over unstructured text corpora**. Its expensive, fragile core — LLM-extract entities and relations from every chunk, cluster into communities, LLM-summarize each community — exists to *manufacture* a graph where none exists.

You are not in that situation:

1. **Your graph already exists**, in your transactional databases. Applications, hires, company hierarchies, job→skill, job→location, recruiter→job. You do not need an LLM to extract it. The single most expensive and least reliable stage of canonical GraphRAG is free for you.
2. **You are doing retrieval and ranking, not generation.** There is no "answer" to synthesize, no community summary to map-reduce, no sensemaking query. Importing Microsoft GraphRAG's indexing pipeline would buy you nothing and cost you a fortune.

So the correct question is not "how do I run GraphRAG?" It is:

> **How do I inject graph structure into an existing dense-retrieval stack without breaking the latency budget?**

That question has good, boring, well-understood answers. The rest of this doc is those answers, ordered by return on engineering effort.

The one piece of GraphRAG literature that transfers directly is the 2025–26 cost-reduction consensus (LazyGraphRAG, LinearRAG, *Towards Practical GraphRAG*): **cheap structure + intelligence deferred to query time beats expensive LLM-built graphs.** You should be even more aggressive about that than the papers are, because your structure is already free.

---

## 1. What the graph actually is

Define it explicitly before writing any code. For a job marketplace:

**Nodes**

| Type | Rough cardinality | Volatility |
|---|---|---|
| Skill (canonicalized, ESCO/O*NET-aligned) | 10⁴–10⁵ | very low |
| CanonicalTitle | 10⁴ | very low |
| Company | 10⁶ | low |
| Location / Market | 10⁴ | very low |
| **Job** | 10⁶–10⁸ | **very high — expires in ~30 days** |
| Seeker | 10⁷–10⁸ | high |

**Edges**

- `Job –REQUIRES→ Skill` — dictionary/NER match against a canonical skill taxonomy. **Not an LLM call per job.**
- `Job –AT→ Company`, `Job –IN→ Location`, `Job –HAS_TITLE→ CanonicalTitle`
- `Skill –CO_OCCURS→ Skill` — PMI over job postings, thresholded
- `Skill –RELATED→ Skill` — from the taxonomy hierarchy
- `CanonicalTitle –CAREER_STEP→ CanonicalTitle` — **mined from seeker resume/employment transitions**
- `Company –SIMILAR→ Company` — industry, size, co-application
- `Seeker –APPLIED / VIEWED / HIRED→ Job`
- `Job –CO_APPLIED→ Job` — derived, thresholded

### The asset embeddings genuinely cannot replicate

The **title transition graph** mined from employment histories. Cosine similarity between JD texts will not tell you that *Actuarial Analyst → Data Scientist* is a common, successful transition, or that *SWE → Engineering Manager* is a real career edge while *SWE → Sales Engineer* is rare. That's behavioral knowledge encoded in your data and invisible to a text encoder. If you build one graph asset, build this one.

The second: **hard constraints.** Work authorization, security clearance, salary band, on-site radius, licensure. Embeddings *blur* these — the whole thesis of Ant Group's KAG is that vector similarity is not logical relevance. A symbolic layer over your graph is the fix. Your biggest early win may be constraint correctness rather than semantic reasoning.

---

## 2. Four injection points, ranked by ROI

You do not need to pick one. Ship them in this order.

### (A) Graph features in the re-ranker — *do this first*

You already have a ranking layer. Add graph-derived features to it. **Zero retrieval-path latency cost, no architectural change, immediately A/B-testable.**

- Personalized PageRank score of the job seeded from the seeker's application history (precomputed per seeker, refreshed daily)
- Shortest path / similarity in the skill graph between query-extracted skills and job-required skills
- Company affinity: has the seeker applied to similar companies?
- Career-step plausibility: is `seeker_current_title → job_title` a known transition edge, and how frequent?
- Co-application count between this job and jobs the seeker engaged with

If these features carry no weight in the model, **you have just cheaply falsified the hypothesis that graph structure helps you** — before building any graph infrastructure. That is the entire point of doing it first.

### (B) Graph-enriched embeddings — *cheapest retrieval-side win*

Keep the retrieval architecture *identical*. One ANN call. Change only what goes into the encoder.

**B1 — Structured text expansion (start here).** When embedding a job, don't embed the raw JD. Embed a normalized document assembled from the graph: canonical title + canonical skills + expanded related skills + company descriptor + seniority + location. Same for the query side: normalize extracted entities before encoding. This is document expansion using your graph as the expansion source, and it requires **no new model and no serving change** — just a different string into the Qwen encoder at index time.

This alone typically recovers a large fraction of what people hope to get from a GNN.

**B2 — Embedding propagation.** Offline, smooth job embeddings over the content graph:

```
e'_j = α · e_j + (1 − α) · mean(e_i for i in Neighbors(j))
```

with neighbors defined by shared canonical title + skill overlap. One hop, α ≈ 0.7–0.85, tuned on held-out applications. Batch job, no serving change, index rebuild only.

⚠️ Propagate over the **content** graph, not the interaction graph, for the retrieval index. Interaction edges don't exist for jobs posted an hour ago, and in a job marketplace freshness is the product.

**B3 — GNN two-tower (PinSage/GraphSAGE-style).** Real gains, real cost: new training pipeline, staleness management, cold-start handling. **Only build this after (A) and (B1) have proven the signal exists.** Do not start here.

### (C) Multi-channel candidate generation with RRF fusion — *the "GraphRAG" proper*

This is the pattern that actually corresponds to graph-augmented retrieval, and it's what *Towards Practical GraphRAG* validates: run channels in parallel, fuse by Reciprocal Rank Fusion.

```
query
  ├─ Channel 1: dense ANN (existing GPU KNN)          → top 500
  ├─ Channel 2: graph expansion from ANN seeds        → top 200
  ├─ Channel 3: query-entity → skill/title expansion  → top 200
  └─ Channel 4: personalized graph walk from seeker   → top 200
                          ↓
              RRF fusion → dedupe → hard-constraint filter → rank
```

- **Channel 2** takes your top ~50 ANN hits as seeds and looks up a *precomputed* neighbor list per job. This is an array lookup, not a graph traversal. Purpose: recover jobs the embedding missed because the JD is written in different vocabulary.
- **Channel 3** parses the query to entities, walks the small skill/title graph, and hits an inverted index `skill → jobs`. Purpose: career-transition and rare-skill-combination queries.
- **Channel 4** is personalization/exploration from the seeker's history.

**Hard-constraint filtering must happen after fusion, not before.** Graph expansion will happily pull in jobs that violate location/authorization constraints.

### (D) Graph-aware query understanding

Parse the query into `{skills, title, seniority, location, modality, constraints}`, canonicalize against the taxonomy, then decide routing. This is where you convert "semantic input" into something you can reason over symbolically — and it's what makes (C) work at all.

---

## 3. The scalability answer

This is the part that dissolves the problem. **Separate your graph into two layers with completely different treatment.**

### Layer 1 — the small, stable entity graph

Skills, canonical titles, companies, locations, and the edges among them. This is on the order of **10⁵–10⁶ nodes**. It fits in memory on a single machine. It changes weekly, not hourly. You can afford *anything* here — PageRank, community detection, embeddings, full traversal, even LLM-assisted curation, because you run it once a week over a small object.

### Layer 2 — the huge, churning job/seeker graph

10⁶–10⁸ nodes with 30-day lifetimes. You **never traverse this at query time.**

### The trick

> Jobs connect to the entity layer with a handful of edges each. Route *all* graph reasoning through the small stable layer, then project back to jobs through an inverted index.

```
job  →  {skills, title, company, location}   ← small, stable, cheap to reason over
                    ↓ traverse here
        {expanded skills, adjacent titles}
                    ↓ inverted index
                  jobs
```

This is exactly LinearRAG's relation-free bipartite construction, arrived at from the systems side rather than the NLP side. Cost is O(entity layer), independent of job count.

### Serving-side consequences

- **Precompute, don't traverse.** Materialize `job → top-50 graph neighbors` as a static CSR array (int32 offsets + int32 neighbors). For 10M jobs × 50 neighbors that's ~2GB — mmap it, or park it on the GPU next to your index. Lookup is a memory read, ~microseconds.
- **Precompute personalized PageRank per seeker** offline (or per seeker *segment* for the tail), refreshed daily. Never run PPR in the request path.
- **New jobs get content edges instantly** (skills/title/company are known at posting time) and interaction edges when they accumulate. This is the correct cold-start story and it falls out of the layering for free.
- **Rebuild cadence:** entity layer weekly; job→neighbor CSR nightly with an incremental append path for new postings; interaction aggregates hourly.

### Latency budget

| Stage | Added latency |
|---|---|
| GPU ANN (existing) | — |
| Channel 2 neighbor lookup (CSR, 50 seeds) | ~1–2 ms |
| Channel 3 inverted-index lookup | ~2–3 ms |
| RRF fusion + dedupe | <1 ms |
| **Total added** | **~5 ms** |

Nothing here requires distributed graph traversal, a graph database in the request path, or an LLM call. If your design needs any of those at serving time, it's the wrong design.

---

## 4. Staged rollout

| Stage | Work | Risk | Decision gate |
|---|---|---|---|
| **0** | Build canonical skill/title taxonomy + entity extraction over jobs (dictionary/NER, no LLM) | Low | Coverage ≥90% of jobs get ≥3 canonical skills |
| **1** | Graph features into existing re-ranker (§2A) | Low | Do the features earn weight? **If no, stop here.** |
| **2** | Graph-enriched job/query text into Qwen encoder (§2B1) | Low | Recall@k on held-out applications, sliced by segment |
| **3** | Channel 3 (skill/title expansion) + RRF (§2C) | Medium | Gains on tail/exploratory queries specifically |
| **4** | Channel 2 (precomputed job neighbors) + Channel 4 (personalized) | Medium | Latency holds; diversity/coverage improves |
| **5** | Embedding propagation (§2B2), then GNN two-tower (§2B3) | High | Only if 1–4 showed real signal |

Stages 0–2 are a few weeks of work and answer the strategic question. Stage 5 is a quarter and should never be attempted first.

---

## 5. Evaluation — and the finding that should shape your rollout

The single most transferable empirical result from the literature (*RAG vs. GraphRAG*, arXiv 2502.11371):

- Graph helps on **multi-hop, comparison, exploratory** queries.
- Vector retrieval **wins by 11–14% on single-hop, detail-oriented** queries.

Mapped to your product:

| Query type | Expect |
|---|---|
| `"software engineer san francisco"` (head, direct) | Graph adds **nothing**. Possibly hurts by diluting precision. |
| `"jobs like my current role but in fintech"` | Graph helps — company/industry edges |
| `"what can I move into from actuarial analyst"` | Graph helps **a lot** — title transition edges; embeddings are weak here |
| Rare skill combinations, sparse markets | Graph helps — expansion recovers thin candidate pools |
| Cold-start seekers | Graph helps via content edges |

**Therefore: route, don't blanket-apply.** The literature's "Selection" hybrid (classify the query, send it down one path) captures most of the benefit at a fraction of the cost of the "Integration" hybrid (always run both). Head queries take the pure ANN path. Tail/exploratory queries get graph channels. Your query-understanding layer (§2D) is the router.

**Offline metrics:** recall@k and NDCG against held-out applications/hires, but **always sliced** — head vs. tail queries, cold vs. warm jobs, dense vs. sparse markets. A flat aggregate number will hide the effect entirely, because head traffic dominates and graph does nothing there.

**Online:** application rate, application→interview rate, and employer-side quality. Watch for the two-sided effect — graph expansion increases applications *per job* concentration if you're not careful.

---

## 6. Failure modes to design against

1. **Popularity feedback loop.** Co-application edges amplify already-popular jobs. Two-sided marketplaces die of this — the top 5% of jobs absorb all applications and employers on the tail churn. Mitigate with degree normalization on the interaction graph and explicit diversity/fairness constraints in fusion.
2. **Cold-start jobs.** A posting from an hour ago has zero interaction edges — and freshness is your product. This is why the retrieval index propagates over the **content** graph only.
3. **Taxonomy drift.** Skill taxonomies rot fast (every framework released this year is missing). Budget for a continuous taxonomy-extension pipeline; this is where a targeted LLM is genuinely worth it — on the small entity layer, weekly, not per job.
4. **Constraint leakage.** Graph expansion pulls in structurally-similar but ineligible jobs. Filter after fusion, always.
5. **Extraction recall is your ceiling.** The GraphRAG literature's most underrated finding is that only ~65.8% of answer entities made it into the constructed KG in one careful study. Your equivalent: what fraction of jobs get correctly canonicalized skills and titles? **Measure this intrinsically**, not just end-to-end. Everything downstream is capped by it.
6. **Fairness/legal exposure.** Graph-based expansion in employment is a regulated surface (EEOC, EU AI Act high-risk classification for employment). Proxy variables propagate readily along company and school edges. Get this reviewed before launch, not after.

---

## 7. If you *do* have an LLM surface

The above is pure retrieval. If you also run conversational job search, "why you match" explanations, or recruiter-side candidate search, then GraphRAG-proper applies — but only as a **context assembly** step:

- Retrieve via §2C, then serialize the **subgraph** around the top candidates (job → skills → seeker's overlapping skills → transition path) into the prompt.
- This gives the LLM the *grounded relational path*, which is what makes explanations faithful instead of plausible-sounding. "You match because you have 6 of 8 required skills, and Actuarial Analyst → Data Scientist is a transition 2,300 seekers on the platform have made" is a graph fact, not an LLM guess.
- Explanation quality is where graph structure pays off most visibly to users, and it's a much lower-risk place to start LLM integration than the retrieval path.

---

## 8. Bottom line

- Do **not** port Microsoft GraphRAG. Its expensive core solves a problem you don't have.
- Your graph is already in your databases. Extract it with dictionaries and SQL, not LLMs.
- Scalability is solved by **routing all reasoning through a small, stable entity layer** and precomputing job-level neighbor lists — not by scaling a graph database.
- Prove the signal exists in your **re-ranker** before you touch the retrieval path.
- **Route by query type.** Graph earns nothing on head queries and a lot on career-transition and sparse-market queries.
