# Job Search Scenarios — Where Retrieval Runs Out

*Design note 8. The job-search-native version of the scenario catalog. Note 7 derives the same territory abstractly; this one starts from what job seekers actually do.*

---

## 0. The reframe: you built one stage out of nine

Job search is not a lookup. It's a **campaign that runs for weeks or months** through distinct stages. Semantic retrieval serves exactly one of them.

| # | Stage | What the seeker is doing | Served by ANN retrieval? |
|---|---|---|---|
| 1 | **Orientation** | "What's out there for someone like me?" | ❌ no query exists yet |
| 2 | **Targeting** | "Which roles should I actually go after?" | ❌ needs feasibility, not relevance |
| 3 | **Discovery** | "Find me those jobs" | ✅ **this is your system** |
| 4 | **Qualification** | "Can I actually get *this* one?" | ❌ per-(user, job) assessment |
| 5 | **Application strategy** | "Which 10 this week?" | ❌ set-level, not item-level |
| 6 | **Tailoring** | Résumé emphasis, cover letter | ❌ generation |
| 7 | **Diagnosis** | "40 applications, zero responses. Why?" | ❌ analysis of their own funnel |
| 8 | **Interview prep** | Company research, likely questions | ❌ different corpora |
| 9 | **Offer evaluation** | "Is this good? Should I negotiate?" | ❌ benchmarking + comparison |

**Eight of nine stages are unserved.** That's the honest framing of the opportunity — not "should we add an agent to search," but "search is one-ninth of the job, and we've built only that ninth."

Most of the eight don't need an agent either. They need different machinery.

---

## 1. Stage by stage

### Stage 1 — Orientation
> *"I've been an actuarial analyst for 6 years in Chicago. What are my options?"*

There is no query. The seeker doesn't know the vocabulary of the roles they'd be good at — that's the whole problem. Returning "actuarial analyst" jobs is answering a question they didn't ask.

**What it needs:** a market map from the transition graph — reachable titles, with observed volumes, comp deltas, and live inventory counts. **No LLM loop.** This is a graph traversal plus aggregation, and it's arguably the single highest-value thing you can build off the graph.

**Who's stuck here:** new grads, career changers, anyone laid off from a shrinking function.

---

### Stage 2 — Targeting
> *"Of those options, which should I actually pursue?"*

Relevance is the wrong metric. The question is **feasibility × payoff**: can I get it, how long would it take, what does it pay, how many exist?

**What it needs:** per-target gap computation against live inventory — the planner (note 6). This is one of only two genuinely agentic stages.

---

### Stage 3 — Discovery ✅
> *"Show me quantitative analyst roles in Chicago."*

Your current system, plus graph query enrichment (note 4). Fast, single-turn, correctly so.

---

### Stage 4 — Qualification
> *"Am I actually competitive for this one? And does it sponsor?"*

Two sub-questions, both per-(user, job), neither answerable by a relevance score:

- **Bar assessment.** "You match 6 of 8 required skills; the gaps are Python and Spark. Applicants who got interviews here typically had 5+ years — you have 6."
- **Hard-constraint verification.** Sponsorship, degree, clearance, on-call, travel. Buried in prose.

**What it needs:** the precomputed criteria matrix (note 5) for the head, LLM verification for the tail. Skill-overlap comparison is pure graph.

**Why this stage matters most:** it's where seekers waste the most time. Forty-five minutes on an application for a role that never sponsored is the single most infuriating experience in job search, and it's entirely preventable with an offline extraction job.

---

### Stage 5 — Application strategy
> *"I have maybe 10 real applications in me this week. Which ones?"*

A ranked list is the wrong output. The seeker needs a **portfolio**: some reach, some match, some safety, spread across employers.

Top-10-by-relevance is frequently ten near-identical stretch roles at three companies. Individually optimal, collectively a bad week — because ten applications to one company is **one bet, not ten**.

**What it needs:** set-level re-ranking (MMR/DPP) over the top-N, with an explicit reach/match/safety banding from the gap computation. **No LLM.** Ship in a sprint.

---

### Stage 6 — Tailoring
> *"What should I emphasize for this one?"*

Genuinely generative, and genuinely useful — but grounded: "this JD weights Spark and streaming heavily; your résumé mentions Spark once, in a bullet about batch jobs." That's a diff between JD-extracted requirements and résumé-extracted evidence, with the LLM writing it up.

**Guardrail:** never invent experience. The output must be *reordering and emphasis of things the seeker actually claims*, never new claims. This is a résumé-fraud vector if you get it wrong, and it's the seeker who bears the consequence.

---

### Stage 7 — Diagnosis
> *"I've applied to 40 jobs in 6 weeks and heard nothing. What am I doing wrong?"*

**Nothing is retrieved from the job corpus.** This is analysis of the seeker's own funnel against a cohort baseline:

- Are they applying above their level? (their YOE vs. the median of what they apply to)
- Is their profile thin on the skills their targets require?
- Are they only applying to the most-contested postings?
- Is application recency bad — always 3 weeks after posting?
- Is a hard constraint silently killing them — no sponsorship, wrong geography?

**What it needs:** a fixed analysis pipeline with LLM narration. Not an agent that "figures it out." Compute the deviations, rank by effect size, explain the top two.

**This is probably the highest emotional-value feature in the entire product**, and it's a deterministic report. Nobody else can build it, because it needs their application outcomes joined to market data.

---

### Stage 8 — Interview prep
> *"I have an interview at Northwestern Mutual Thursday."*

Needs corpora you may not have: company news, Glassdoor-style signal, interview reports, the team's stack. Cross-source retrieval, and the one stage where classic RAG-over-documents is the right shape.

Partly out of scope unless you own or license those sources.

---

### Stage 9 — Offer evaluation
> *"They offered $165k. Is that good? Should I push?"*

Comp benchmarking against title × level × geography × industry, plus their alternatives in flight.

**What it needs:** structured aggregation over your own data. You have comp distributions nobody else has at this granularity. **No agent** — this is a query and a chart.

⚠️ Keep it **descriptive, not prescriptive**: "the 60th percentile for this title in Chicago insurance is $172k" — not "you should ask for $180k." Negotiation advice creates liability you don't want, and the descriptive version is more useful anyway.

---

## 2. Seeker situations that break retrieval hardest

Stage isn't the only axis. Certain situations break the retrieval model itself.

**The career changer.** Their résumé vocabulary doesn't match target JDs at all. An actuarial analyst reads nothing like a data science JD, even though the transition is common and works. **This is the case where embeddings fail worst and your graph helps most** — and it's a retrieval fix (query enrichment, note 4), not an agent.

**The visa-constrained seeker.** Sponsorship is invisible in your index, so they waste effort at a catastrophic rate — possibly the majority of their applications are structurally void. Highest-ROI single verification criterion you can extract. Fix this before anything else in Class 4.

**The laid-off seeker with a severance clock.** High urgency, high anxiety, needs volume and triage. **Do not put a 20-second agent in front of this person.** They need fast results, a portfolio view, and diagnosis when it isn't working. Speed is a feature with emotional weight here.

**The returner with a career gap.** Needs employers who don't auto-screen on gaps — an attribute that exists nowhere in your index and is genuinely hard to verify. Best available proxy: employers who *have historically hired* people with gaps, computable from your own hire data. A good example of the graph answering something no JD text can.

**The over-leveled senior.** Inventory is thin; their failure mode is empty result sets, not too many. This is where the constraint lattice and binding-constraint explanation matter most: *"3 Director roles in Chicago fintech; 14 if you include hybrid."*

**The new grad.** No history, cold start, and — more fundamentally — no idea what roles exist. Stage 1 problem masquerading as a stage 3 problem.

---

## 3. Observable failure moments — go find these in your logs

Every scenario above shows up as a measurable event. These are instrumentable **today**, before you build anything:

| Failure moment | Log signature | Fix | Stage |
|---|---|---|---|
| Zero results after filters | `results == 0` post-filter | Constraint lattice | 4 |
| Ten thousand results, all alike | high result count, low intra-set diversity, low CTR | MMR/DPP | 5 |
| Applied and heard nothing | ≥20 applications, 0 responses, 30d | Diagnosis report | 7 |
| Wasted a great application | applied → later marked "no sponsorship" | Criteria matrix | 4 |
| Same jobs every day | high repeat-impression rate in feed | Saved search + diff | — |
| Can't re-find last week's job | search → no click → repeat query | Session history/recall | — |
| Is this even real? | high impression, near-zero application on old postings | Listing validity check | 4 |
| Query reformulated 3+ times | reformulation count in session | Clarification chips | 1/3 |
| Searches own current title only | query ≈ profile title, low CTR | Orientation surface | 1 |

**Rank your roadmap by the volume of these events, not by which feature is most interesting to build.** You already have the data to do that ranking this week.

---

## 4. The recruiter side

Half the marketplace, and much more agentic-friendly — high latency tolerance, complex intents, paid seats.

- **Sourcing** — "senior backend, fintech, open to Chicago, likely to move" — complex boolean plus propensity modeling
- **Slate construction** — needs a *varied* shortlist, not fifteen near-identical profiles. Set optimization again, inverted.
- **Market sizing** — "how many candidates with this profile exist, and what would we have to pay?" Pure aggregation, constant demand.
- **Req calibration** — "this req has been open 90 days; what's blocking it?" Diagnosis, inverted: usually the requirements are unrealistic for the comp band, and you can *prove* it from your data.
- **Outreach at volume** — Class 7 action, with all the guardrails.

**Req calibration is the sleeper.** Recruiters get told "no candidates exist"; you can show them exactly which requirement is binding and what relaxing it yields. Same constraint-lattice machinery as the seeker side, pointed the other way, sold to the side that pays.

---

## 5. What to build, in job-search terms

| Priority | Build | Serves | LLM loop? |
|---|---|---|---|
| 1 | Sponsorship + degree + clearance extraction → filters | Stage 4, visa-constrained seekers | No |
| 2 | Career map from transition graph | Stage 1, changers, new grads | No |
| 3 | Portfolio banding (reach/match/safety) + diversity re-rank | Stage 5 | No |
| 4 | Saved search + new-since-last-visit diff | Whole campaign | No |
| 5 | "Why no responses" diagnostic report | Stage 7 | Narration only |
| 6 | Constraint lattice / binding-constraint explanation | Stage 4, senior seekers | No |
| 7 | Comp benchmarking | Stage 9 | No |
| 8 | Req calibration (recruiter side) | Paid side | No |
| 9 | Career planner | Stage 2 | **Yes** |
| 10 | Tailoring assistant | Stage 6 | **Yes**, guarded |

**Items 1–8 need no agent.** They need extraction, graph traversal, set re-ranking, and aggregation — and together they cover seven of the nine stages.

The two genuinely agentic items are last, and both work better once 1–8 exist to ground them.
