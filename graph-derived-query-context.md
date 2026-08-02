# Graph-Derived Textual Context for Query-Side Semantic Retrieval

*Design note 2. Setup: `retrieve(query + context)` with a Qwen bi-encoder over a GPU ANN index. The query is given; the **context** is what the graph produces, as text.*

---

## 0. Why this is the hard version of the problem

The framing is right, and it's the right place to put the graph. But "walk the graph, verbalize it, append it to the query" fails in a specific, well-documented way, and the whole design has to be organized around that failure.

### The central problem: query drift in a single-vector bi-encoder

Your encoder pools a token sequence into one vector (mean-pool or last-token). Pooling is (approximately) an *average*. If the query is 4 tokens and the graph context is 150 tokens, the pooled vector sits essentially on the context centroid. **You have not enriched the query; you have replaced it.**

This is not a hypothetical. It's the same wall that classical pseudo-relevance feedback (Rocchio) and its dense successors (ANCE-PRF, Vector-PRF, ColBERT-PRF) all hit: expansion adds recall on some queries and destroys precision on others, and the aggregate is often a wash. The literature that actually governs your design is **dense query expansion / PRF and graph-to-text verbalization** — not GraphRAG. GraphRAG's retrieval stage is mostly about traversing to *find* content; your problem is about *encoding* content you've already found without drowning the query.

Three consequences that shape everything below:

1. **Query weight must be an explicit, tunable parameter**, not an emergent property of token counts.
2. **Context length is a budget**, not a bag. More graph facts is monotonically worse past a fairly small threshold.
3. **Changing query-side input distribution is a train/serve mismatch.** Your encoder's query tower has an implicit prior over what queries look like. Feeding it a 150-token verbalized subgraph moves it off-distribution — off-manifold relative to what your job vectors were indexed against.

---

## 1. What the context should actually contain

Before verbalization, be clear about what job the context is doing. In a job marketplace there is one dominant answer:

> **Graph context is a vocabulary translation layer between seeker language and employer language.**

Seekers write *"data scientist"*, *"want to move into ML"*, *"something like my current job but better pay."* Job descriptions are written by employers in HR/req vocabulary, 600 tokens long, full of stack names, level codes, and compliance boilerplate. The bi-encoder is being asked to bridge a **length asymmetry** (4 tokens vs 600) and a **vocabulary asymmetry** (self-description vs. requisition) at the same time.

Your graph is the bilingual dictionary between those two registers. That's the actual value — more than "multi-hop reasoning," which is the framing GraphRAG papers push and which mostly doesn't apply to you.

So the context should be built to make the query *look structurally like a job posting*. Concretely, for query `"data scientist"` with seeker profile `{title: Actuarial Analyst, skills: [SQL, R, GLM, statistics], exp: 4y, loc: Chicago}`:

| Context component | Graph operation | Yields |
|---|---|---|
| **Skill closure** | seed skills → PMI/taxonomy neighbors, top-k by edge weight | Python, pandas, scikit-learn, predictive modeling, A/B testing |
| **Title neighborhood** | canonical title → co-occurring / substitutable titles | Applied Scientist, Quantitative Analyst, Decision Scientist |
| **Transition context** | `Actuarial Analyst –CAREER_STEP→ Data Scientist`, with support count and **bridging skills** (skills present in target, absent in source) | "commonly requires adding Python and machine learning" |
| **Seniority mapping** | (years, source title) → target level band | Data Scientist II / Senior |
| **Market context** | location × title → dominant industries, common employer types | insurance, fintech, healthcare analytics in Chicago |
| **Requisition register** | title → most frequent JD phrasings for this role family | "build predictive models", "partner with stakeholders" |

That last row is the one people skip and it may be the highest-value: mine the **actual phrasing distribution** of JDs for the target role family from your own corpus, and inject the top discriminative phrases. That is a graph-mediated lookup (title → job set → phrase distribution) that directly closes the vocabulary gap.

### Verbalize the delta, not the union

If the query already says "data scientist," putting *"data scientist"* in the context is not free — it consumes budget and reweights a term you already have. **Spend the context budget on what the query does not say.** Query-term weight is handled separately and explicitly (§3), not by repeating terms into the context.

### Rank facts before verbalizing

You will always have more graph facts than budget. Rank by:

```
utility(fact) ∝ edge_weight(seed → fact)          # PPR or PMI
              × discriminativeness(fact)           # IDF over the job corpus
              × (1 − redundancy(fact, query))      # don't re-say the query
```

`discriminativeness` matters: "communication skills" appears in 60% of JDs and carries no retrieval signal; "GLM" or "Databricks" narrows the candidate pool hard. Naive top-k-by-edge-weight will hand you the generic ones, because generic skills have the highest degree in a co-occurrence graph. **Degree-normalize or you will systematically verbalize the least useful facts.**

---

## 2. Verbalization: subgraph → text

The format matters more than people expect, because your encoder is a language model.

**Do not linearize raw triples.** `(Data Scientist, requires, Python) (Data Scientist, requires, SQL) ...` is off-distribution for an LM trained on prose and pools badly. Empirically, across the KG-to-text literature (WebNLG and successors), natural-language rendering beats triple linearization for downstream LM encoders.

**Three formats, in increasing cost:**

**(a) Templated prose — start here.** Deterministic, auditable, zero latency, no LLM.

```
Target role: Data Scientist (also Applied Scientist, Quantitative Analyst).
Mid-level, 4 years experience, Chicago.
Existing skills: SQL, R, statistical modeling, GLM.
Typically required: Python, machine learning, predictive modeling, A/B testing.
Common in insurance, fintech, and healthcare analytics.
```

**(b) LLM-verbalized, cached.** Same facts, rendered fluently by a small model. Better encoder alignment; only viable because it's cacheable (§5).

**(c) Graph-conditioned HyDE — the strong version.** Have an LLM generate a *synthetic ideal job posting* from the query + subgraph, then embed that.

This is worth taking seriously, because it attacks both asymmetries at once: the synthetic doc is JD-length and JD-register, so you are now doing **doc-to-doc retrieval** instead of query-to-doc. HyDE's original result was that a hypothetical document embeds closer to real relevant documents than the query does; grounding the generation in your graph fixes HyDE's main weakness (the LLM hallucinating the wrong specifics), because the skills, titles, and market facts come from your data rather than the model's priors.

Cost is an LLM call (~200–500ms), so it's for personalized feeds, saved searches, and "explore" surfaces — not interactive typeahead. And it is entirely cacheable per (canonical query, seeker segment).

---

## 3. Fusion: how context and query combine

This is where you control drift. Five options.

### Option 1 — Concatenate, single encode
`v = E(query ⊕ context)`

Simplest, one ANN call. **But query weight is implicit in token counts**, which is exactly the failure mode. If you do this, you need an explicit weighting device:
- **Query repetition** (the query2doc trick): repeat the query *n* times so it holds a fixed share of the token budget. Crude, effective, tune *n*.
- **Instruction prefix**: `"Query: {q}\nCandidate background: {context}"` — Qwen embedding models are instruction-tuned and respond to role markers; this recovers some structure that flat concatenation destroys.

Verdict: viable, but you're steering a continuous quantity with a discrete hack.

### Option 2 — Dual encode + weighted vector fusion ⭐ *start here*

```
v = normalize( α · E(query_with_profile) + (1 − α) · E(graph_context) )
```

Two short encoder passes (cheap — queries are short and you're on GPU), one ANN call.

Why this is the right first move:
- **α is an explicit dial.** Tune it, and tune it *per query class* (§6).
- **α = 1 is an exact no-op.** Your current system is a strict special case, so the A/B is clean and the rollback is free.
- **No encoder retraining required to get a first read.** You are composing in vector space with a frozen model.

Limitation: a weighted sum is a crude approximation of joint encoding — it can't represent "Python *in the context of* actuarial background." Accept that for v1; it buys you a measurement.

### Option 3 — Multi-vector retrieval + RRF ⭐ *best quality, and cheap on your hardware*

Encode the query and each context *aspect* separately; run k ANN probes; fuse by RRF.

```
v_q      = E(query + profile)              → top 300
v_skill  = E(skill-closure context)        → top 200
v_trans  = E(transition context)           → top 200
v_market = E(market/seniority context)     → top 200
                     ↓ RRF
```

- **Zero dilution.** Every aspect is expressed at full strength in its own probe.
- **Debuggable and auditable.** You know which aspect surfaced each result — which matters in a regulated employment surface, where "why was this shown" is a question you may have to answer formally.
- **Per-aspect weighting** falls out of the RRF weights.
- You're on GPU: 4 query vectors in a batch instead of 1 is close to free. Batched ANN is throughput-bound, not latency-bound, at these sizes.

This is the same multi-channel structure from the previous doc, but with the channels defined in **embedding space over verbalized graph context** rather than by graph traversal. That's the version that fits your framing.

### Option 4 — Late interaction (ColBERT-style)
Token-level matching would handle the length asymmetry natively. Too large a change to your index. Note it and move on.

### Option 5 — Learned context gating
Small model predicts α (or per-aspect weights) from query features. This is the *end state* of Option 2/3, once you have data on when expansion helps. Don't start here — you need the logs from Options 2/3 to train it.

---

## 4. The train/serve mismatch — the thing that will bite you

Your job vectors were indexed by an encoder that saw JD-shaped input. Your query vectors came from query-shaped input. That asymmetry is baked into the index.

Introducing graph context shifts the **query-side input distribution** while leaving the index fixed. Query vectors move into a region of the space they didn't previously occupy. Three responses:

**(a) Frozen encoder + vector fusion (Option 2).** Sidesteps it — you never feed the encoder off-distribution input, you just move within the span of two in-distribution vectors. This is a real advantage and the reason to start there.

**(b) Retrain the query tower on graph-augmented inputs.** The proper fix. Positives from applications/hires. **The hard negatives are the crux:** mine negatives from jobs that graph expansion *surfaces but seekers don't apply to*. Without those, expansion teaches the model to over-generalize — it learns that anything skill-adjacent is relevant, and precision collapses on head queries. Random negatives will not catch this because random negatives are trivially separable.

**(c) Expand both sides symmetrically.** See §5.

---

## 5. The doc-side counter-argument — take it seriously

There's a real case that graph context belongs on the **job side**, not the query side, or at minimum on both.

| | Query-side expansion | Doc-side expansion (doc2query-style) |
|---|---|---|
| Compute budget | Bounded by latency SLA | **Offline, unbounded** |
| Query drift | **The central problem** | Doesn't exist |
| Pooling distortion | Severe (short input) | Mild (JDs are already long) |
| Personalization | **Yes** — seeker-specific | No — one expansion serves everyone |
| Cache complexity | High | Trivial (it's the index) |

The honest answer is that they do different jobs:

- **Doc-side graph expansion** handles *vocabulary normalization*. Enrich each job with canonical skills, expanded skill closure, canonical title synonyms, and level normalization at index time. This narrows the vocabulary gap for **every** query, costs nothing at serving, and cannot drift.
- **Query-side graph context** handles *personalization and intent* — the career-transition path, the seeker's existing skills, the market they're in. This genuinely cannot be precomputed into the index because it depends on who's asking.

**Do doc-side first.** It's offline, it's safe, it has no latency cost, and it raises the floor for everything else. Then query-side context is doing only the work that *must* be done at query time, which means it needs a smaller budget, which means less drift. These compound in the right direction.

---

## 6. Serving: caching is what makes the expensive options affordable

The naive read is "graph walk + verbalize + encode, per request." Don't do that.

**Cache the context *embedding*, not the context text.** If you cache text you still pay the encoder; if you cache the vector, serving is a lookup plus vector arithmetic.

Exploit the two independent axes of concentration:

- **Query axis:** canonicalized queries are enormously head-concentrated. Precompute context vectors for the top ~100k canonical queries; that covers the large majority of traffic. Tail queries fall back to α = 1 (pure query) — which is fine, and is also where you'd learn whether tail coverage is worth extending.
- **Seeker axis:** a seeker's profile-derived context is stable across sessions. Compute it nightly, or on profile edit. Never in the request path.

Composition then costs a cache hit on each axis plus a weighted sum:

```
v = normalize( α·v_query + β·v_seeker_ctx + γ·v_querycontext_ctx )
```

which is microseconds. Cold path (both misses) falls back to α = 1 and warms asynchronously. This also makes graph-conditioned HyDE (§2c) viable — an LLM call you make once per canonical query per day is a completely different cost object than one per request.

---

## 7. Evaluation, including a direct drift diagnostic

**Ablation ladder** (each row adds one thing; measure recall@k and NDCG, always sliced by head/tail query and cold/warm job):

1. query only
2. \+ seeker profile *(presumably your current system)*
3. \+ skill closure
4. \+ title neighborhood
5. \+ transition context
6. \+ market/seniority
7. \+ doc-side expansion (orthogonal — cross it with the above)

**Measure drift directly.** For each query, compute

```
drift = cosine( E(query), v_fused )
```

Plot the distribution. Queries below ~0.7 have had their meaning changed, not enriched — pull those and read them. In my experience this bucket is where the precision losses concentrate, and it's usually a small, identifiable class (short ambiguous queries where the profile context overwhelms an explicit intent: seeker searches `"product manager"` and gets engineering roles because their profile is engineering-heavy). That specific failure — **context overriding explicit stated intent** — is the one users notice and complain about, because they typed something and got something else.

**α should not be global.** Expect the tuned optimum to look roughly like:

| Query class | α (query weight) |
|---|---|
| Explicit, specific (`"senior python engineer remote"`) | ~0.9 — context adds little, risks precision |
| Short, ambiguous (`"analyst"`) | ~0.6 — context disambiguates |
| Exploratory (`"what can I do next"`) | ~0.3 — context *is* the query |
| Empty query / feed | 0 — pure context retrieval |

That last row is worth noticing: **an empty query with full graph context is your recommendation feed.** The same machinery covers search and discovery, which is a good architectural reason to build it this way.

---

## 8. What this architecture cannot do

Be clear-eyed about the ceiling, because it determines what stays symbolic.

1. **Negation and hard constraints.** "No relocation," "does not require clearance," "excluding contract roles." You cannot express negation in an additive embedding — adding "clearance" to the context pulls you *toward* clearance jobs. Constraints stay in a filter layer, always applied post-retrieval. This is exactly KAG's thesis: vector similarity is not logical relevance.
2. **Exact numeric conditions.** Salary bands, radius, years-of-experience thresholds. Filters, not embeddings.
3. **Composition of the wrong facts.** Verbalized context has no error bars. If your skill extractor mislabels a job, the false skill enters the context as confidently as a true one and propagates into every expansion. Extraction precision on the entity layer is the ceiling on everything here — measure it intrinsically.
4. **Recency.** A verbalized graph context is a static string; job freshness is a ranking signal, not a retrieval-context signal.

---

## 9. Recommended path

1. **Doc-side graph expansion** into the job index. Offline, safe, no drift, raises the floor for everything else.
2. **Option 2** (dual encode, weighted fusion) with **templated prose** context. Frozen encoder, α tunable, α = 1 is an exact no-op. This is a clean A/B that answers "does graph context help at all" in weeks.
3. **Per-query-class α**, driven by your query-understanding layer. Most of the remaining win.
4. **Option 3** (multi-vector + RRF) once you know which aspects earn their keep — cheap on GPU, removes drift, and gives you per-aspect attribution.
5. **Retrain the query tower** on graph-augmented inputs with expansion-mined hard negatives. This is where the step change is, and it requires the logs from steps 2–4.
6. **Graph-conditioned HyDE** for feed/explore surfaces, cached per canonical query per day.

Steps 1–2 are the measurement. Step 5 is the payoff. Don't invert them.
