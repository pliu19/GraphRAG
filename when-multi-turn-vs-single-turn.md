# When You Need Multi-Turn / Agentic Search vs. Single-Turn Retrieve-and-Rank

*Design note 4. Context: job marketplace, GPU ANN retrieval + ranking, graph-enriched queries.*

---

## 0. The wrong framing to avoid

The industry default right now is "put an LLM agent in front of search." That is almost always a downgrade for the traffic that actually matters, because:

| | Single-turn | Agentic |
|---|---|---|
| Latency | 100–300 ms | 2–8 s |
| Cost/query | ~$0 | 10²–10³× |
| Determinism | Full — debuggable funnel | Non-deterministic; A/B and root-cause both get much harder |
| Auditability | Feature attribution | Prompt archaeology |

That last row is not a small thing for you. Employment matching is a regulated surface. "Why was this job shown to this person" is a question you may have to answer formally, and a deterministic ranker answers it far more comfortably than an LLM planning loop.

So the question isn't "should we be agentic." It's **which specific query classes are architecturally unserveable by single-turn**, and route only those.

---

## 1. The precomputation test — apply this first

Most things that look like they need an agent are actually a graph join you haven't precomputed yet.

> **Before building an agent for a multi-hop query, ask: does the hop live on the stable entity layer? If yes, precompute it and stay single-turn.**

Worked case: *"what can I move into from actuarial analyst?"*

This looks agentic — it's a reasoning question, multi-hop, requires knowing career structure. But the hop is `title → CAREER_STEP → title`, entirely on the entity layer, stable over months. Precompute it into the transition table and it becomes a **single-turn lookup with α = 0.3**. No agent, 200ms, deterministic, auditable.

**Multi-hop retrieval ≠ multi-turn agent.** A graph collapses multi-hop into single-hop *at index time*. That is the entire point of having built one. If you build an agent to traverse a graph you could have precomputed, you've paid 1000× to do offline work online.

The hop genuinely requires runtime iteration only when it depends on **intermediate results computed from the query itself** — not on stable structure.

---

## 2. The five conditions that actually force multi-turn

Single-turn retrieve-and-rank works when: *the query (after enrichment) determines a region of the corpus, and the user's job is to select from that region.* It breaks under exactly these conditions.

### (1) The output is not a set of items

```
"How many senior ML roles in Chicago pay above $200k?"
"What's the median comp for my title in fintech vs. healthcare?"
"Which skills show up most in jobs where I get rejected?"
```

The answer is a **computation over the result set**, not a selection from it. A ranker returns items; it structurally cannot return a number, a comparison, or a distribution. Genuine capability gap.

⚠️ But this needs **structured query generation** (text-to-SQL/aggregation over your job attributes), *not* a free-form agent. Cheaper, faster, and far more reliable. Don't reach for an agent when you need a query compiler.

### (2) A required input doesn't exist yet and must be elicited

Job seekers have **constructed preferences, not retrieved ones**. They don't know they care about remote until they see the commute; don't know their comp expectations until they see the band; don't know they'd consider a smaller company until they see one.

Single-turn cannot access this information because *it doesn't exist at query time* — it's created by the interaction. This is the classic conversational-recommender preference-elicitation problem, and it's the cleanest genuine gap in the list.

### (3) Steps have runtime dependencies that can't be precomputed

```
"Find jobs I could get with 3 months of upskilling, and tell me what to learn."
```

Decomposes into: infer current skill profile → find reachable roles → compute per-role skill gap → filter gaps to ~3 months of effort → rank by gap × desirability → map gaps to resources.

Each step consumes the previous step's output. The gap computation depends on *this seeker's* skills against *each candidate role* — a per-(user, job) computation that can't be precomputed across the cross-product. Genuine planning.

### (4) Results must be verified against criteria absent from your index

```
"Biotech roles that sponsor H1B and don't require a PhD, within 30 min of Cambridge."
```

Visa sponsorship and degree requirements are usually buried in JD prose, not in your schema. So: retrieve → read the JD → verify → if too few survive, relax a constraint and re-retrieve. That's an agentic loop, and it's **common in job search precisely because JDs are unstructured** and many decision-critical attributes never made it into your fields.

This is probably your highest-value agentic use case, because it's the one where users currently give up.

### (5) Ambiguity is disjoint *and* items are expensive to scan

`"analyst"` spans financial / data / business / policy / actuarial — disjoint career paths. Diversifying the slate gives 4 of each and makes the user do the disambiguation by reading 600-token JDs.

⚠️ Note this is a **UX cost tradeoff, not a capability gap** — diversification *can* solve it. One clarifying question is just cheaper than 20 wasted impressions. Which means the fix is a clarification chip, not an agent (§3, tier 2).

---

## 3. Four tiers, not two

The single-turn / agentic binary is the source of most bad architecture here. There are four, and the middle two are badly underrated.

### Tier 1 — Single-turn retrieve + rank
**~95% of seeker traffic.** Head queries, navigational queries, graph-enriched exploratory queries, similar-jobs. Everything in the enrichment spec.

### Tier 2 — Stateful multi-turn refinement (deterministic, no LLM)
Faceted refinement, clarification chips, and **critiquing**: *"like that one but more senior"*, *"closer to home"*, *"same field, bigger company"*.

This is multi-turn but **not agentic** — it's a state machine over filters plus re-retrieval, seeded by the previous result. Latency stays at Tier 1. Most of the practical value people attribute to "conversational search" lives right here, and it costs nearly nothing.

If you build one thing beyond Tier 1, build this.

### Tier 3 — Structured query generation
Aggregation, comparison, market questions. LLM compiles the natural-language query into a structured query over job attributes; the query executes deterministically. Bounded, verifiable, cacheable. Handles condition (1).

### Tier 4 — Agentic planning loop
Multi-step plans with runtime dependencies (condition 3), and retrieve-verify-relax loops over unstructured JD text (condition 4). Rare, slow, expensive, high value per query.

---

## 4. Routing — predict *and* escalate on failure

Two mechanisms, both cheap. Use both.

**Predictive routing** (extends the α router you already need):

| Signal | Route |
|---|---|
| Aggregation lexicon — *how many, average, compare, most common, median* | Tier 3 |
| Advice/planning framing — *what should I, how do I get into, help me* | Tier 4 |
| Session follow-up referencing prior results — *that one, more like, but cheaper* | Tier 2 |
| Otherwise | Tier 1 |

**Reactive escalation** — this is the better mechanism, because it costs nothing until you actually fail:

```
Tier 1 runs
   ├─ < N results survive filtering        → Tier 4 constraint-relaxation loop
   ├─ top-k scores below threshold         → Tier 2 clarification
   ├─ result set spans disjoint clusters   → Tier 2 disambiguation chips
   └─ user reformulates ≥2× in session     → Tier 2/4 escalation
```

**Escalate on observed failure, not on predicted intent.** Intent classification on short queries is unreliable and you'll misroute head traffic into 5-second responses. Failure signals are unambiguous and you only pay for them when Tier 1 has already lost.

---

## 5. Where to build agentic first: the recruiter side

If you're going to invest in Tier 3/4, **build it on the employer/recruiter side before the seeker side.** The economics are not close:

| | Seeker side | Recruiter side |
|---|---|---|
| Query volume | Very high | Low |
| Latency tolerance | Seconds feel broken | Minutes are normal |
| Monetization | Free / ad-supported | **Paid seat** |
| Intent complexity | Mostly simple | Genuinely complex boolean + market research |
| Aggregation need | Occasional | **Constant** — "how many candidates with X in Y?" |
| LLM cost justification | Hard | Trivial |

Recruiters already iterate, refine, and do market sizing — the multi-turn interaction pattern exists in their workflow whether or not you support it. Seekers mostly want to scan a list fast. Conversational job search has repeatedly underdelivered on the seeker side for exactly this reason: **users don't want to chat, they want to scan.**

---

## 6. Measurement — the part that quietly breaks

You cannot A/B a multi-turn system against a single-turn one with query-level metrics. NDCG@10 on turn 1 of a clarification flow will look *worse* than single-turn while the overall experience is better.

Switch the unit of analysis:

| Single-turn metric | Multi-turn equivalent |
|---|---|
| NDCG@k, recall@k | **Session success rate** (application / save / hire) |
| CTR | Turns-to-first-application |
| Zero-result rate | Session abandonment rate |
| — | Escalation rate, and success rate *given* escalation |

Track escalation success separately. If Tier 4 fires on 2% of sessions and resolves 60% of them, that's a strong result even if it's invisible in aggregate query metrics.

---

## 7. Summary

**Stay single-turn when** the query determines a region of the corpus and the user selects from it. Enrichment plus graph precomputation covers far more of this than people assume — including most "reasoning" queries like career-transition exploration.

**Go multi-turn when** the output isn't a set of items, a needed input must be elicited, steps have runtime dependencies, results need verification against unindexed criteria, or disjoint ambiguity would waste the whole slate.

**Apply the precomputation test first.** If the hop is on the stable entity layer, it belongs in the graph, not in an agent.

**Build Tier 2 before Tier 4.** Deterministic stateful refinement captures most of the practical value of "conversational search" at Tier 1 latency and cost.

**Start agentic on the recruiter side**, where latency tolerance is high, intents are genuinely complex, and someone is paying per seat.
