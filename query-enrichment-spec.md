# Query-Side Graph Enrichment — Implementation Spec

*Concrete design. Assumes: Qwen bi-encoder, GPU ANN over job vectors, entity layer (skills / titles / companies / locations) built per design note 1.*

---

## 0. The organizing idea: slot filling, not graph search

Don't run a generic graph walk and verbalize whatever comes back. A PPR blob over a heterogeneous graph gives you an undifferentiated ranked list that you then have to turn into prose — and you'll have no control over what ends up in it.

Instead: **define the output template first, then define one graph lookup per slot.**

This gives you, for free:
- every slot is a **key → value fetch**, not a traversal (so it's precomputable and O(1) at serving)
- slot-level ablation in evaluation
- slot-level budgets, so nothing can crowd out anything else
- readable, auditable context — which you need in a regulated employment surface

---

## 1. Query understanding → two seed sets

```
raw_query + seeker_profile
        ↓
┌──────────────────────────────────────────────┐
│ canonicalize:                                │
│   gazetteer + fuzzy → skill nodes            │
│   title normalizer   → canonical title       │
│   geo resolver       → market node           │
│   attribute tagger   → seniority, modality,  │
│                        employment type       │
│   negation detector  → negated spans         │
└──────────────────────────────────────────────┘
        ↓
  S_q  = seeds from the QUERY      → intent
  S_u  = seeds from the PROFILE    → background
  C    = hard constraints          → filter layer, NOT context
  N    = negated spans             → DROP entirely
```

**Three rules that matter operationally:**

1. **Keep `S_q` and `S_u` separate.** Never merge into one seed pool. Query seeds are *intent*; profile seeds are *background*. Merging them is exactly how you get "seeker types `product manager`, gets engineering roles because their profile is engineering-heavy."
2. **Constraints leave the context.** Location, remote, employment type, salary band, clearance → filter layer. Anything you can filter exactly, you should not be embedding.
3. **Strip negated spans before verbalization.** `"no relocation"` must never reach the context string — an additive embedding cannot represent negation, so verbalizing it pulls you *toward* relocation jobs. This is a real bug you will ship if you don't handle it explicitly.

---

## 2. Slot inventory

Each row is one precomputed table. All keys live on the small entity layer.

| Slot | Key | Graph operation | Budget |
|---|---|---|---|
| `target_titles` | canonical title | title–title substitutability, top-k by co-application PMI | 2–4 |
| `expanded_skills` | skill (union over seeds) | skill–skill PMI, degree-normalized | 8–12 |
| `bridging_skills` | (profile_title, target_title) | requirement profile of target **minus** seeker's skills | 3–5 |
| `transition_evidence` | (profile_title, target_title) | `CAREER_STEP` edge + support count | 1 clause |
| `seniority_band` | (years_exp, profile_title, target_title) | level mapping table | 1 term |
| `industry_context` | (target_title, market) | job-corpus aggregation | 2–3 |
| `jd_phrasing` | target_title | top discriminative n-grams from JDs of that title family | 3–5 |

**`bridging_skills` is a set difference, not a walk** — the skills the target role requires that the seeker doesn't have. It's the single most query-specific slot and the one that most directly encodes "what this person is trying to become."

**`jd_phrasing` is the vocabulary bridge.** Mine the actual phrasing distribution of real postings for that title family from your own corpus. This is what converts seeker register into requisition register, and it's usually the highest-yield slot per token spent.

### Sizing

All keys are entity-layer objects. Titles ~10⁴, skills ~10⁵. Even `(source_title, target_title)` pairs only have meaningful support for ~10⁵–10⁶ combinations. Every table fits comfortably in a KV store or in memory. **Nothing here scales with job count.**

---

## 3. Within-slot ranking

```python
def utility(fact, query_tokens):
    w = edge_weight(seed, fact)              # PMI / normalized co-occurrence
    if not (DF_LO <= doc_freq(fact) <= DF_HI):
        return 0                             # band filter, see below
    if near_match(fact, query_tokens):
        return 0                             # verbalize the delta, not the union
    return w
```

**Use a document-frequency band, not an IDF weight.** `DF_LO ≈ 0.1%`, `DF_HI ≈ 20%` of the job corpus.

- Above the band: `"communication skills"`, `"team player"` — appear everywhere, carry zero retrieval signal.
- Below the band: extraction noise — typos, one-off internal tool names, garbage from bad JD parses.

Pure IDF weighting *rewards* the bottom of that range, so in a noisy extraction pipeline it will systematically surface junk. The band is the fix.

**Degree-normalize edge weights.** Generic skills have the highest degree in a co-occurrence graph, so raw top-k-by-weight hands you exactly the facts the band filter is trying to remove. Belt and braces.

---

## 4. Verbalization

**Write the context in job-description register, not seeker register.** You are matching against indexed JDs, so the context should read like a posting, not like a cover letter. No `"I have..."`, no `"seeking..."`.

**v1 template (single string, ~60–90 tokens):**

```
{target_title} role at {seniority_band} level.
Related titles: {target_titles[1:]}.
Skills: {expanded_skills}.
Commonly also requires {bridging_skills}.
Typical in {industry_context}.
Work involves {jd_phrasing}.
```

Templated prose, deterministic, zero latency, auditable. Start here — do not start with an LLM verbalizer.

**v2 aspect grouping** (for the multi-vector variant in §5):

| Aspect | Slots | Retrieval behavior |
|---|---|---|
| **A — Role** | `target_titles`, `seniority_band`, `jd_phrasing` | what the job *is* |
| **B — Skills** | `expanded_skills`, `bridging_skills` | what it *requires* |
| **C — Trajectory** | `transition_evidence`, `industry_context` | why it fits *this seeker* |

---

## 5. Fusion

### v1 — dual encode, weighted sum

```
v = normalize( α · E(query ⊕ profile) + (1−α) · E(context) )
```

One ANN call. Frozen encoder. **α = 1 reproduces your current system exactly**, so the A/B is clean and rollback is free.

### v2 — multi-vector + RRF

```
v_q  = E(query ⊕ profile)     → top 300
v_A  = E(aspect_A)            → top 200
v_B  = E(aspect_B)            → top 200
v_C  = E(aspect_C)            → top 200
              ↓ RRF (weighted)
     dedupe → hard-constraint filter → rank
```

4 batched query vectors on GPU is close to free. Zero dilution, plus per-aspect attribution — you know which aspect surfaced each result.

### α routing

Rules first, learned later. Router inputs are already computed in §1.

| Condition | α |
|---|---|
| Explicit canonical title + ≥1 constraint | 0.9 |
| Explicit canonical title, no constraints | 0.75 |
| Skills only, no title | 0.6 |
| Short / ambiguous (< 3 content tokens) | 0.6 |
| Exploratory intent detected | 0.3 |
| **Empty query** | **0.0** ← this is your feed |

That last row is worth designing for deliberately: empty query + full graph context *is* the recommendation feed. Search and discovery become the same code path with one parameter difference.

---

## 6. Serving and caching

**Cache the context *embedding*, not the context text** — otherwise you still pay the encoder per request.

| Cache | Key | Size | Refresh |
|---|---|---|---|
| `qctx_vec` | canonical query | ~100k × 4KB ≈ **400 MB** | weekly |
| `uctx_vec` | **seeker segment** | ~50k × 4KB ≈ **200 MB** | nightly |

**Cache profile context at segment level, not per seeker.** Cluster seekers by (canonical title, skill cluster, seniority band, market) into ~50k segments. Per-seeker caching is 40GB+ and unnecessary — retrieval only needs to get you into the right neighborhood.

This gives a clean division of labor that falls straight out of the cost math:

> **Retrieval personalizes at segment granularity. Ranking personalizes at individual granularity.**

Serving cost is then two hash lookups plus a weighted sum — microseconds.

**Fallbacks:**

| Situation | Behavior |
|---|---|
| Cache miss (either axis) | α = 1, serve immediately, warm async |
| No canonical title matched | Drop role slots, skills-only context |
| No profile | α = 1, or query-only expansion |
| Constraint-heavy query | α → 0.9, lean on the filter layer |

---

## 7. Worked example

**Query:** `"data scientist"`
**Profile:** Actuarial Analyst · 4 yrs · SQL, R, GLM, statistics · Chicago

```
S_q = {title: Data Scientist}
S_u = {title: Actuarial Analyst, skills: [SQL, R, GLM, statistics],
       years: 4, market: Chicago}
C   = {market: Chicago}          → filter
N   = {}
```

Slot fills:

```
target_titles       → Applied Scientist, Quantitative Analyst, Decision Scientist
expanded_skills     → Python, machine learning, predictive modeling,
                      pandas, scikit-learn, A/B testing, data pipelines
                      (dropped: "communication", df > 20%)
bridging_skills     → Python, machine learning         (in target, not in profile)
transition_evidence → Actuarial Analyst → Data Scientist, 2,300 observed
seniority_band      → mid / II
industry_context    → insurance, fintech, healthcare analytics
jd_phrasing         → "build predictive models", "partner with stakeholders",
                      "productionize models"
```

Context string:

> *Data Scientist role at mid level. Related titles: Applied Scientist, Quantitative Analyst, Decision Scientist. Skills: Python, machine learning, predictive modeling, pandas, scikit-learn, A/B testing. Commonly also requires Python and machine learning for candidates from actuarial backgrounds. Typical in insurance, fintech, and healthcare analytics. Work involves building predictive models, partnering with stakeholders, and productionizing models.*

Router: explicit title + 1 constraint → **α = 0.9**... but note this query is short and the seeker is mid-transition, so this is exactly the case to A/B α = 0.75 against.

`normalize(0.9·E(query⊕profile) + 0.1·E(context))` → ANN → filter `market = Chicago` → rank.

Without enrichment, this query retrieves Data Scientist postings that lexically resemble the phrase. With it, it also reaches postings written as *Quantitative Analyst* at insurance firms requiring GLM and Python — jobs this seeker is unusually well-positioned for and would never have seen.

---

## 8. Evaluation

**Slot-level ablation.** Add one slot at a time; keep what earns its budget:

```
base:  query + profile
 +1:   target_titles
 +2:   expanded_skills
 +3:   bridging_skills
 +4:   transition_evidence
 +5:   industry_context
 +6:   jd_phrasing
```

Metrics: recall@k / NDCG against held-out applications and hires, **always sliced** by head-vs-tail query and cold-vs-warm job. Head traffic will dominate any aggregate and hide the effect entirely.

**Drift diagnostic — build this on day one:**

```
drift = cosine( E(query ⊕ profile), v_fused )
```

Plot the distribution. Anything below ~0.7 has had its meaning changed rather than enriched. Read those queries. The failure concentrates in one identifiable class — short explicit query, strong contradicting profile — and that's the class users complain about, because they typed something and got something else.

**Guardrail metrics:** zero-result rate, result-set concentration (Gini over jobs shown), and the share of impressions going to cold jobs. Expansion tends to concentrate traffic; watch it.

---

## 9. Build order

| # | Work | Gate |
|---|---|---|
| 1 | Canonicalization + slot tables (offline) | ≥90% of queries resolve ≥1 seed |
| 2 | Templated verbalizer + drift diagnostic | — |
| 3 | Dual-encode fusion, global α, shadow traffic | drift distribution sane |
| 4 | Slot ablation, fix budgets | which slots earn their tokens |
| 5 | Rule-based α routing | gains hold on tail without head regression |
| 6 | Multi-vector + RRF | per-aspect attribution, no dilution |
| 7 | Retrain query tower w/ expansion-mined hard negatives | the step change |
| 8 | Learned α | last |

Steps 1–4 are a few weeks and tell you whether any of this is worth it. Step 7 is where the real gain is, and it needs the logs from 3–6 to build the negative set.
