# Scenario Catalog — When Agentic Search Beats Single-Turn Retrieval

*Design note 7. Companion to note 4, which gave the decision framework. This one enumerates the actual scenarios, derived systematically rather than listed ad hoc.*

---

## 0. Derivation: what single-turn retrieval assumes

Rather than listing scenarios by intuition, start from the assumptions baked into retrieve-and-rank. Each broken assumption generates a scenario class.

| # | Assumption | Class when it breaks |
|---|---|---|
| A1 | The information need is fully expressed in the query + profile | **Elicitation** |
| A2 | The answer is a subset of the corpus | **Computation** |
| A3 | Items can be scored independently (pointwise relevance) | **Set optimization** |
| A4 | All decision-relevant attributes are in the index | **Verification** |
| A5 | One retrieval pass suffices — no dependencies between steps | **Planning** |
| A6 | The need is satisfied at a point in time | **Monitoring** |
| A7 | The system only reads; it does not act | **Action** |

Seven classes. Below: the scenarios in each, why single-turn fails, and — critically — **the cheapest thing that actually works**, which for most classes is not an agent.

---

## Class 1 — Elicitation (A1 breaks: the need isn't in the query)

### Scenarios

**1.1 Constructed preferences.** The seeker doesn't know they care about remote until they see the commute, or about company size until they see a 12-person startup next to a 40,000-person bank. The preference doesn't exist at query time; the interaction creates it.

**1.2 Cold start, no profile.** New user, empty history, two-word query. There's nothing to enrich with and nothing to personalize on.

**1.3 Disjoint ambiguity.** `"analyst"` spans financial / data / business / policy / actuarial — disjoint career paths, not shades of one intent. Diversification gives four of each and makes the user do the disambiguation by reading 600-token JDs.

**1.4 Goal underspecification.** *"I want something better."* Better how — comp, title, hours, stability, growth? All five are retrievable; the query specifies none.

### Why single-turn fails
It can only hedge (diversify the slate). That works when items are cheap to scan. Job descriptions are not.

### Cheapest thing that works
**Tier 2, deterministic.** Clarification chips + diversified fallback. `"Which kind of analyst work? [Data] [Financial] [Business] [Show all]"` — one click, no LLM, Tier 1 latency. Entropy over unspecified facets tells you which facet to ask about.

**Only escalate to an agent for 1.4**, where the dimensions themselves need to be inferred from context.

---

## Class 2 — Computation (A2 breaks: the answer isn't a subset of the corpus)

### Scenarios

**2.1 Aggregation.** *"How many senior ML roles in Chicago pay above $200k?"*

**2.2 Distribution / benchmarking.** *"Is my current salary competitive for my title and years of experience?"*

**2.3 Segment comparison.** *"Fintech vs. healthcare for my title — comp, growth, remote availability?"*

**2.4 Self-diagnostic.** *"I've applied to 40 jobs and gotten no responses. What's wrong?"* — this is analysis over the **user's own funnel** (application targets, profile completeness, level mismatch, skill gaps) compared against a cohort baseline. Nothing is retrieved from the job corpus at all.

**2.5 Offer evaluation.** *"Is this offer good?"* — needs comp benchmarking, company signals, and the user's alternatives, joined.

### Why single-turn fails
A ranker returns items. It structurally cannot return a number, a distribution, a comparison, or a causal account.

### Cheapest thing that works
**Tier 3 — structured query generation, not an agent.** LLM compiles the natural-language question into a query over your job attributes; the query executes deterministically. Bounded, verifiable, cacheable, fast.

⚠️ **2.4 is the exception and the most valuable one.** Self-diagnosis needs a fixed analysis pipeline (compute funnel stats → compare to cohort → identify the largest deviation → explain) with the LLM only narrating. Build it as a deterministic report with LLM presentation, not as an agent that "figures it out."

---

## Class 3 — Set optimization (A3 breaks: items aren't independent)

**This is the class most commonly missed, and rankers are structurally incapable of it.**

### Scenarios

**3.1 Application portfolio.** A seeker has ~10 applications of realistic effort per week. They need a **mix** — some reach, some match, some safety. Top-10-by-relevance can easily be ten near-identical stretch roles at one company. Individually optimal, collectively terrible.

**3.2 Correlated-risk hedging.** Ten applications to one company is one bet, not ten. Diversification across companies, industries, and geographies is a set-level property that pointwise relevance cannot see.

**3.3 Effort budget allocation.** Applications aren't free — a tailored cover letter costs 30 minutes. Which 10 of 200 matches justify the effort, given probability of response × payoff?

**3.4 Recruiter slate construction.** Same problem inverted: a recruiter needs a *slate* — varied backgrounds, not fifteen copies of the same profile.

### Why single-turn fails
Ranking is pointwise or pairwise. Portfolio construction optimizes a **set-level objective** (expected value under a budget, with covariance between items). Different mathematical object.

### Cheapest thing that works
**Deterministic set re-ranking — MMR or DPP over the top-N.** No LLM. This covers 3.1–3.2 and 3.4 well, and it's a well-understood technique you can ship in a sprint.

**3.3 needs a response-probability model**, which is an ML problem, not an agent problem.

Genuinely agentic only when the user wants to *negotiate the portfolio interactively* — "more safety, less reach."

---

## Class 4 — Verification (A4 breaks: attributes aren't indexed)

### Scenarios

**4.1 Attributes buried in prose.** Visa sponsorship, security clearance, degree requirement, on-call expectations, travel percentage. Present in JD text, absent from your schema.

**4.2 Negation.** *"Roles that don't require on-call rotation."* You cannot express this in an embedding — adding "on-call" to the query pulls you *toward* on-call jobs.

**4.3 Cross-source verification.** *"Is this company stable?"* needs news, funding data, layoff reports, and reviews — none of which live in your job index.

**4.4 Listing validity.** *"Is this actually still open?"* Ghost jobs are a real and growing problem; freshness of the posting isn't the same as validity of the req.

### Why single-turn fails
The filter doesn't exist, so the constraint silently becomes a soft semantic signal — which means it isn't enforced at all.

### Cheapest thing that works
**Precompute the criteria matrix** (note 5, §4). The criteria distribution is heavily head-concentrated — sponsorship, degree, remote, clearance, YOE, contract-vs-FTE probably cover 80%+ of demand. Extract them offline for all active jobs, promote them to filterable fields, and most of this class collapses into Tier 1.

The agent handles only the **tail** criterion — the one nobody anticipated — plus 4.3, which genuinely requires retrieval from other corpora.

---

## Class 5 — Planning (A5 breaks: steps have runtime dependencies)

### Scenarios

**5.1 Career path planning.** *"What's my path into ML engineering?"* — see note 6. Gap is per-(user, target); path choice depends on objective and budget.

**5.2 Skill-gap → learning plan.** *"What should I learn to double my options in 3 months?"* — requires computing the marginal option-value of each skill against live inventory.

**5.3 Conditional retrieval.** *"If there's nothing decent in Boston, show me remote roles at the same level."* Step 2's existence depends on step 1's outcome.

**5.4 Rare-constraint intersection.** *"Rust + medical devices + Series B or later."* ANN returns noise because no single vector covers the conjunction. Decompose, retrieve each component, intersect, verify.

**5.5 Multi-hop joins over live data.** *"Companies that hired someone with my exact background in the last year, and what they have open now."* Transitions → companies → open reqs. The first hop depends on *this user's* background.

### Why single-turn fails
No single query vector expresses a dependency. This is the one class with **no precomputation escape hatch**, because the intermediate results depend on the query itself.

### Cheapest thing that works
Apply the precomputation test first — 5.1's *transition edges* are precomputable even though the plan isn't.

Then: **out-of-band planner that emits durable state** (note 6). Not in the query path. 5.3 and 5.4 are cheap enough to run inline; 5.1–5.2 should be asynchronous session-level surfaces.

---

## Class 6 — Monitoring (A6 breaks: the need persists over time)

### Scenarios

**6.1 Standing queries.** *"Tell me when a Staff-level role opens at any of these 15 companies."*

**6.2 Change detection.** *"What's new since I last looked?"* — relevance is defined by **delta**, not absolute score. A job you already dismissed has zero value even at rank 1.

**6.3 Market drift.** *"Is demand for my skill set declining?"* — a time-series question over the corpus.

**6.4 Application state tracking.** Where each application stands, what's gone quiet, what needs follow-up.

### Why single-turn fails
The unit isn't a query, it's a **subscription**. Scoring is relative to a previously-seen set.

### Cheapest thing that works
**Saved search + set diff. Deterministic, no LLM.** This class gets over-agentified more than any other — people build "monitoring agents" for what is a cron job plus a set difference against seen-items.

6.3 is Tier 3 (a time-series aggregation). Only 6.4 benefits from an agent, and only for drafting follow-ups.

---

## Class 7 — Action (A7 breaks: the system takes side effects)

### Scenarios

**7.1 Delegated application.** *"Apply to everything matching these criteria."*

**7.2 Tailored artifact generation.** Per-job cover letters, résumé emphasis reordering.

**7.3 Recruiter outreach at volume.** Personalized sequences to a candidate slate.

**7.4 Scheduling and response handling.** Recruiter emails, interview slots.

### Why single-turn fails
Retrieval is read-only by definition. These have **irreversible side effects**.

### What's required
This is the **highest-risk class** and the one where "agentic" is unavoidable — but it needs guardrails no other class does:

- **Verification before action, always.** Never auto-apply to a job whose hard constraints haven't been verified.
- **Confirmation gates**, batched to stay usable. Review-then-approve, not silent execution.
- **Rate limits.** Mass auto-application degrades the marketplace for everyone and gets seekers blacklisted by employers.
- **Full audit trail.** Every action, its trigger, and its approval.
- **Reversibility where it exists**, explicit warning where it doesn't.

Two-sided caution: an auto-apply feature that works well for seekers can destroy the employer side of your marketplace. Model the equilibrium before shipping it, not after.

---

## Summary: what each class actually needs

| Class | Cheapest sufficient solution | Genuinely needs an LLM loop? |
|---|---|---|
| 1 Elicitation | Clarification chips + diversification | Rarely (only 1.4) |
| 2 Computation | Structured query generation | No — 2.4 needs a fixed pipeline |
| 3 Set optimization | MMR / DPP re-ranking | No |
| 4 Verification | Precomputed criteria matrix | Tail only (+ 4.3) |
| 5 Planning | Out-of-band planner | **Yes** |
| 6 Monitoring | Saved search + set diff | No |
| 7 Action | — | **Yes**, with heavy guardrails |

**Five of seven classes have a deterministic solution that captures most of the value.** Only planning (5) and action (7) genuinely require an LLM in a loop.

That's the honest engineering read: the perceived need for agentic search is mostly a need for **set-level re-ranking, precomputed attributes, structured query compilation, and saved-search diffs** — four unglamorous things, all of which are faster, cheaper, auditable, and testable.

Build those first. Whatever demand survives is your real agentic requirement, and it will be a much smaller and better-specified surface than it looks like today.

---

## Prioritization for a job marketplace

| Priority | Build | Class | Why |
|---|---|---|---|
| 1 | Precomputed criteria matrix | 4 | Highest volume of user frustration; converts to Tier 1 |
| 2 | Saved search + diff | 6 | Job search is a campaign; cheapest real feature here |
| 3 | Portfolio/diversity re-ranking | 3 | Structurally missing today; pure win, no LLM |
| 4 | Clarification chips | 1 | Cheap, high-frequency |
| 5 | Structured query generation | 2 | Unlocks a whole question type |
| 6 | Self-diagnostic report | 2.4 | High perceived value, fixed pipeline |
| 7 | Out-of-band planner | 5 | The real agent; needs 1–6 to be grounded |
| 8 | Delegated action | 7 | Last. Marketplace-equilibrium risk. |
