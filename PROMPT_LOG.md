# PROMPT_LOG — Query Fan-Out Engine: Prompt Iteration Journal

This document records the full prompt engineering process for `POST /api/fanout/generate` — what I observed at each iteration, what I changed, and why.

For the broader narrative of how I approached the assignment (research, tooling decisions, the threshold sweep), see [`my_journey/how_i_made_records.md`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/my_journey/how_i_made_records.md).

---

## Background — What the Prompt Has to Do

The fan-out engine must decompose any user query into 10–15 sub-queries spanning 6 fixed types (`comparative`, `feature_specific`, `use_case`, `trust_signals`, `how_to`, `definitional`), return them as a strict JSON object with no extra fields, and work equally well across completely different domains.

The hard parts:
- **Consistency** — the same structural rules (≥2 per type, 10–15 total) must hold across every query
- **Natural phrasing** — sub-queries should read like real search-engine input, not like essay headings
- **Reliable JSON** — no markdown fences, no prose wrapping, no extra keys

---

## Iteration 1 — First Draft

### The Prompt

> Full prompt text lives in the `_SYSTEM_PROMPT` constant in [`app/services/fanout_engine.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/app/services/fanout_engine.py).
> Iteration 1 was identical to the final prompt except it had **no word-count constraint** and included the instruction: _"Follow the structure of the example exactly."_

### What I Observed

Running `python -m optimization.prompt_tuning.run_prompt_eval` and examining the output:

```
# From iteration_1_too_long_subqueries.csv — target query: "best AI writing tool for SEO"

use_case:
  - AI writing tools for SEO content creation for blogs      ← "for Y for Z" chaining
  - best AI writing software for e-commerce SEO

trust_signals:
  - reviews of AI writing tools for SEO effectiveness
  - case studies on AI writing tools improving SEO rankings   ← sentence, not a search query
```

Two problems:
1. **"X for Y for Z" chaining** — sub-queries like `"AI writing tools for SEO content creation for blogs"` read like run-on sentences, not search queries. A real user would type `"AI writing tools for SEO blogs"`.
2. **Sentence fragments masquerading as queries** — `"case studies on AI writing tools improving SEO rankings"` is a phrase structure, not how anyone types into Google.

Full output log: [`optimization/prompt_tuning/logs/iteration_1_too_long_subqueries.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/prompt_tuning/logs/iteration_1_too_long_subqueries.csv)

### What I Changed

Added two new hard constraints to the prompt:

```
4. Keep each query concise: 4–9 words maximum. No redundant prepositions or filler
   words (avoid patterns like "X for Y for Z" — collapse to "X for Y Z" instead).
5. Write queries the way a real user would type them into a search engine — natural,
   direct, no unnecessary repetition.
```

Also **removed** the instruction to follow the example structure exactly — that instruction was causing the model to mirror the verbose phrasing of the example more than it was enforcing type coverage.

---

## Iteration 2 — Conciseness Fixed, Trust Signals Still Wrong

### What I Observed

The "X for Y for Z" chaining was gone. Query lengths were much better. But one type still looked off:

```
# From iteration_2_trust_signals_not_good.csv — target query: "best AI writing tool for SEO"

trust_signals:
  - AI writing tool for SEO user reviews         ← generic, academic phrasing
  - case studies of AI writing for SEO           ← still not a real search query

# target query: "best CRM software for startups"

trust_signals:
  - CRM software startup user reviews
  - case studies of successful CRM implementations  ← nobody searches this
```

The structural scores were still 1.0 (valid JSON, all types present, ≥2 per type) — so the prompt was passing its automated checks. But reading the output, the `trust_signals` queries were unconvincing. Nobody types "case studies of successful CRM implementations" into a search engine. These read like content-marketing research, not user intent.

Full output log: [`optimization/prompt_tuning/logs/iteration_2_trust_signals_not_good.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/prompt_tuning/logs/iteration_2_trust_signals_not_good.csv)

### Root Cause

The example in the prompt used:
```json
{"type": "trust_signals", "query": "project management software reviews from remote-first companies"},
{"type": "trust_signals", "query": "Monday.com vs Asana case studies remote work 2025"},
```

Both examples lean on words like "reviews" and "case studies" — which are reasonable but signal a specific academic flavour. The model was latching onto that pattern and reproducing it across every domain.

I researched what `trust_signals`-style queries actually look like in the wild — real SERP patterns, Reddit threads about "how do people search for software recommendations". The common vocabulary is more social: "used by", "popular among", "trusted by teams", "most-used at companies like...".

### What I Changed

Replaced the `trust_signals` example queries in the prompt with:

```json
{"type": "trust_signals", "query": "project management tools used by large teams"},
{"type": "trust_signals", "query": "most popular PM software among engineering teams"},
```

Also updated the type description from:
> `trust_signals — reviews, testimonials, case studies, awards, proof points`

to:
> `trust_signals — social proof: used by teams/companies, most popular among professionals, trusted by industry`

The description change anchors the model to social-proof vocabulary (adoption, popularity) rather than academic vocabulary (reviews, case studies).

---

## Iteration 3 — Final Prompt ✅

### What I Observed

```
# From final_iteration_kept_results.csv — target query: "best AI writing tool for SEO"

trust_signals:
  - most trusted AI writing tools for SEO      ← natural, social-proof phrasing
  - popular AI writing software among marketers ← how a person actually searches

# target query: "best CRM software for startups"

trust_signals:
  - most popular CRM for startup sales teams
  - CRM tools trusted by early-stage companies
```

All five test queries scored composite = 1.0. More importantly, the `trust_signals` queries now sound like real user intent rather than research prompts.

Full output log: [`optimization/prompt_tuning/logs/final_iteration_kept_results.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/prompt_tuning/logs/final_iteration_kept_results.csv)

---

## Final Prompt — Design Rationale

The final prompt (in [`app/services/fanout_engine.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/app/services/fanout_engine.py)) uses a **system message + user message** split:

- **System message** contains the role, task, type definitions, hard constraints, JSON schema, and a worked example
- **User message** is just `"Target query: {target_query}"` — minimal, so the model's full attention stays on the system-level instructions

**Why `temperature=0.2`?** Lower temperature reduces hallucinated field names and unusual type identifiers. Some creativity is still useful (we want varied sub-query phrasing, not 12 near-identical queries), so 0.0 would be too rigid. 0.2 has been a reliable default for structured-output tasks in practice.

**Why an explicit JSON schema in the prompt?** Without it, models often wrap the array in different keys (`"queries"`, `"results"`, `"output"`) across calls. Showing the exact schema removes that ambiguity. The code also handles the envelope (`{"sub_queries": [...]}`) defensively in `parse_llm_response()` — if the model adds a wrapper, we unwrap it rather than crashing.

**What I'd do with more time:** Replace the static hardcoded example with a small per-industry example bank. The current example is a project management query — for a health or finance query, the model might lean too heavily on the PM-flavoured phrasing. Dynamic example selection (embed the user's query, find the closest example) would generalise better. See the "What I'd Improve" section in [`README.md`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/README.md#6-what-id-improve-with-more-time).
