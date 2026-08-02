# Tier 4 — The Planning Agent

*Design note 6. This is condition (3) from note 4: multi-step goals with runtime dependencies. Note 5 covered condition (4), the verify-relax loop — a different and easier problem.*

---

## 0. What this actually is

Tier 4 is not search with extra turns. **Its output is not a list of jobs.** It's a plan over a trajectory:

```
"Find jobs I could get with 3 months of upskilling, and tell me what to learn."
"I want to become an ML engineer — what's my path?"
"I'm getting no responses. What's wrong and what do I do about it?"
"What's the fastest route to $200k in my field?"
```

The user is not asking you to retrieve. They're asking you to **search a space of possible futures and return a route.** The job listings show up at the *end*, as evidence that the route is real.

---

## 1. Why this genuinely can't be precomputed

Note 1 established the discipline: if the hop is on the stable entity layer, precompute it. The transition graph (`title → CAREER_STEP → title`) *is* precomputed, and that's what makes "what can I move into?" a Tier 1 lookup.

Planning breaks the test for a specific reason:

| Component | Precomputable? |
|---|---|
| Transition edges | ✅ entity layer, stable |
| Skill requirements per target | ✅ entity layer |
| **Gap = required(target) − has(user)** | ❌ per-(user, target) |
| **Feasible time = Σ gap-closure under user's budget** | ❌ per-(user, target, budget) |
| **Path choice A→B→C vs A→C** | ❌ depends on constraints + objective |
| **Path ranking** | ❌ depends on user's objective weights |

You're searching over **paths**, scored by a **user-specific objective**, under **user-specific constraints**. The cross-product is `users × goals × budgets × objectives × paths`. That's not a table you can build.

This is the real justification for Tier 4, and it's much stronger than the verification-loop justification, because verification has a precomputation escape hatch (note 5, §4) and this does not.

---

## 2. The architecture that makes it cheap: get it out of the query path

The mistake would be putting the planner in front of search. Don't. **The planner is an out-of-band, session-level surface that emits durable state, and that state conditions Tier 1.**

```
     ┌──────────────────────────────────┐
     │  Tier 4 PLANNER                  │   slow (10–30s), rare,
     │  runs on explicit invocation     │   async, rich
     │  or on a trigger                 │
     └──────────────┬───────────────────┘
                    │ emits
                    ▼
          GoalSpec + PathPlan  ────────┐   durable, persisted, editable
                                       │
     ┌─────────────────────────────────▼──┐
     │  Tier 1 RETRIEVAL                  │   fast (200ms), constant,
     │  now goal-conditioned              │   every session
     └────────────────────────────────────┘
```

Consequences, all good:

- **Latency stops being a problem.** The planner isn't in an interactive loop. 20 seconds is fine for something you invoke once and revisit for weeks.
- **Cost stops being a problem.** Once per user per few weeks, not once per query.
- **It compounds.** The plan is state. Every subsequent fast query, every feed impression, every alert gets conditioned on it. The expensive thing is amortized across hundreds of cheap interactions.
- **It degrades safely.** No plan → Tier 1 behaves exactly as today.

The plan becomes a first-class product object: visible, editable, revisable. Not a chat transcript.

---

## 3. The six stages

### A — Goal formalization

Natural language → typed spec. This is the one thing the LLM is uniquely good at here.

```python
GoalSpec:
    origin:      {title, skills, yoe, comp, industry, location}   # from profile
    target:      {titles: [...]} | {attributes: {comp: >180_000}} | UNDERSPECIFIED
    budget:      {months: ?, paycut_tolerance: ?, relocate: ?, study_hrs_wk: ?}
    objective:   minimize_time | maximize_comp | maximize_success_prob | balanced
    constraints: {visa, location, industry_exclusions, ...}
```

The `?` fields are the point. Most are unstated, and every one materially changes the answer.

### B — Reachability expansion

From `origin`, traverse the transition graph to depth 2–3, pruned by constraints. Each node is a state `(title, seniority, industry)`. Cheap — it's the small entity layer, precomputed edges, runtime traversal.

Output: candidate target states + the paths reaching them.

### C — Per-path gap computation *(the part that can't be precomputed)*

For each path, compute against **this user**:

| Gap | Source |
|---|---|
| Skill gap | `required(target) − has(user)`, weighted by learnability |
| Experience gap | YOE delta, domain exposure |
| Credential gap | degree, certification, license |
| **Estimated time** | Σ gap-closure hours ÷ user's study budget |
| **Market depth** | live count of open jobs at target state, under constraints |
| **Observed support** | how many people actually made this move |

### D — Multi-objective scoring → present the Pareto set

Don't return "the best path." The user's objective weights are unknown and they'll recognize what they want when they see it. Compute the Pareto frontier over (time, comp delta, success probability, market depth) and return **three representative paths**:

```
FAST      Actuarial Analyst → Quantitative Analyst
          ~6 weeks (add: Python) · comp +8% · 47 open roles · 2,300 observed

CEILING   Actuarial Analyst → Data Scientist → ML Engineer
          ~14 months · comp +45% · 12 open at step 1 · 890 observed step 1

SAFE      Actuarial Analyst → Senior Actuarial Analyst
          ~in-role promotion · comp +15% · 130 open · highest observed rate
```

Three labeled options beats one recommendation and beats a wall of ten.

### E — Ground every path in live inventory

**This is the link back to your existing stack, and it's what keeps the whole thing honest.**

For each candidate target state, run Tier 1 retrieval with the state's enriched query and the user's hard constraints, and count what comes back.

- "Learn Python and become an ML engineer" — worthless, generic, a chatbot could say it.
- "47 Quantitative Analyst roles are open right now that accept your exact profile; 12 more ML Engineer roles open up if you add Python plus one portfolio project" — actionable, specific to your marketplace, and impossible for anyone without your index.

A path with zero live inventory is not a plan, it's a fantasy. **Prune paths with no market depth**, and say so when you do.

### F — Persist and condition

Write `GoalSpec + PathPlan` to durable state. Then:

- Feed and alerts become goal-conditioned (boost target-state roles and stepping-stone roles)
- The α-router (note 3) reads the goal: an exploratory query from a user with an active plan gets context built from the *plan's* target seeds, not just their current title
- Progress tracking: gap closes as the user adds skills; re-plan on material profile change
- Re-plan triggers: profile edit, 30 days elapsed, market shift in target state

---

## 4. The graph is the reasoning substrate — this is the entire defensibility argument

An LLM has priors about careers. They are generic, US-centric, often wrong, and identical to what every competitor's LLM will say. *"To become a data scientist, consider a master's degree"* is worthless.

Your graph has **your marketplace's observed reality**: who actually moved where, what those roles actually required, what's actually open, in this city, in this industry, this quarter.

> **The LLM plans and narrates. The graph supplies ground truth about what transitions occur and what they cost.**

Enforce this as an architectural rule: **the planner may not assert a transition, a requirement, or a duration that isn't backed by a graph lookup.** Same discipline as the evidence-span rule in note 5. Every number in the output traces to an edge weight, an aggregation, or a live count.

Without that rule you've built a generic career-advice chatbot. With it, you've built something structurally unavailable to anyone who doesn't have your data.

---

## 5. The `ASK` policy inverts here

Note 5 said: ask sparingly, cap at 2, users abandon. **That's correct for search and wrong for planning.**

A user who asks *"what's my path into ML?"* is requesting a consultation. They **expect** to be interviewed. Asking about time budget, pay-cut tolerance, and relocation isn't friction — it's the product working. A planner that doesn't ask feels shallow.

| | Search (Tier 1–2) | Planning (Tier 4) |
|---|---|---|
| Asking reads as | friction | competence |
| Budget | ≤2 per session | 4–6 is fine |
| Format | chips, skippable | structured intake, can be a form |
| Silence means | proceed with defaults | you'll produce a bad plan |

Practical consequence: much of stage A can be a **form**, not a conversation. A short structured intake (time budget, comp floor, relocation, study hours/week) is faster for the user *and* more reliable for you than extracting the same fields from free text over four turns. Use the LLM for the parts a form can't capture — the actual goal, in the user's own words.

---

## 6. Survivorship bias — the honesty problem you must solve

Your transition graph records **people who succeeded and stayed on your platform.** It does not record:

- people who attempted the transition and failed
- people who succeeded but left the platform
- people who never attempted because it was obviously infeasible

So *"2,300 people moved from Actuarial Analyst to Data Scientist"* does **not** mean the move is easy. It means 2,300 succeeded, out of an unknown denominator.

The fix is computable from data you already have:

```
attempt_rate  = seekers with origin title who APPLIED to target-title roles
success_rate  = of those, how many were HIRED into target-title roles
                (or: reached a target-title role within N months)
```

That's a base rate, and it's dramatically more honest than a raw edge weight:

> *"Of actuarial analysts who applied to data science roles, 18% landed one within a year. Those who did were 3× more likely to list Python."*

Report base rates, not just support counts. This also makes the plan genuinely useful — the differentiating skill among successful transitioners is exactly the advice the user wants, and it's a computable contrast between the succeeded and failed cohorts.

---

## 7. Evaluation — be honest that this is hard

The true outcome is 6–12 months out and the counterfactual is unobservable. Don't pretend otherwise. Layer proxies:

| Horizon | Metric |
|---|---|
| Immediate | Plan completion rate; edit rate (high edits = bad formalization) |
| Days | Application rate to on-path roles vs. off-path |
| Weeks | Profile updates matching recommended gap closure; plan revisits |
| Months | Reached a target-state role; comp delta at next role |
| Guardrail | Fraction of asserted facts backed by a graph lookup (should be 100%) |

**The specific risk is building something that feels insightful and does nothing.** Career advice is unusually good at producing satisfaction without producing outcomes. Stage E — live inventory grounding — is your main defense, because a plan tied to 47 actual open roles is falsifiable in a way that "develop your Python skills" is not.

Run a holdout. Even a coarse one.

---

## 8. Risks specific to this tier

1. **Prescriptive vs. descriptive framing.** *"You should get a master's"* is advice with consequences in a regulated domain. *"Of people who made this move, 18% succeeded; 70% of them had Python, 12% had a master's"* is a description of your data. Descriptive framing is more defensible **and** more useful. Enforce it in the output contract, not just the prompt.
2. **Survivorship bias.** §6. Unaddressed, you will systematically over-recommend paths that look popular because they're visible.
3. **Fairness.** If the planner recommends systematically different paths to different demographics, that's an employment-steering problem. Log every path recommendation with its input state and audit the distribution against protected-class proxies. This is the highest-legal-risk surface in everything we've discussed — it's the one that actively tells people what to do.
4. **Staleness.** A 14-month plan built on today's skill demand will be partly wrong. Version plans, timestamp them, re-plan on market drift in the target state.
5. **Feedback loop.** Recommending a path drives applications along it, which strengthens the edge, which increases recommendations. Damp it: score paths on base rates and market depth, not on recommendation-inflated traffic.

---

## 9. Build order

| # | Work | Note |
|---|---|---|
| 1 | Transition base rates (attempt → success), not just edge counts | Fixes §6 before you ship anything on top of it |
| 2 | Gap computation: `required(target) − has(user)` + learnability table | The core primitive; nothing works without it |
| 3 | Structured intake form for `GoalSpec` | Deterministic, no LLM, covers most of stage A |
| 4 | Reachability + gap + **live inventory count** per path | Stages B/C/E — this is already a shippable product with no LLM at all |
| 5 | Pareto selection + three labeled paths | Stage D |
| 6 | LLM: free-text goal parsing + narration, under the graph-backing rule | Stage A/F polish |
| 7 | Goal-conditioned feed and α-router | Where the compounding value is |

**Steps 1–5 are a working planner with no LLM in the loop.** Form in, graph traversal, gap math, live counts out. The LLM enters at step 6 to handle goals a form can't express and to narrate — which is exactly the role it should have, and exactly where it can't invent facts.

Step 7 is where this stops being a feature and starts making your whole retrieval stack better.
