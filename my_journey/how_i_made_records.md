# How I Built This — Thought Process & Decisions

## Starting Out

Read the problem fully two or three times and spent an hour or so researching AEO and GEO before writing any code — wanted to understand what "answer engine optimisation" actually means in practice before making design calls.

Used Copilot to scaffold the AEO module structure (without checks), then reviewed and ran it to get a baseline working before going deeper.

![Initial outputs](some_outputs_c2.png)

---

## Feature 1 — AEO Checks

### Test-first approach

My plan: define all the edge cases first, write them into tests, then implement the checks to pass them. This forced me to think about behaviour before code.

The AI-generated starting point had a single 50+ case file for Check A. I split it into one file per function (`test_has_hedge_phrase.py`, `test_count_words.py`, etc.) with a separate `*_full_integration.py` combining everything on top. Easier to locate failures and reason about what broke.

### Things I caught and fixed

**First paragraph extraction** — the generated code skipped paragraphs with fewer than 3 words. I removed that. For AEO scoring, a too-short opening paragraph is itself a signal worth flagging — skipping it would silently hide a real problem.

Also removed logic that skipped `<p>` tags inside headers, keeping it simple for the assignment scope.

**Plain-text paragraph splitting** — had a minor ambiguity about what "first paragraph break in plain text" means. Used ChatGPT with a few examples to clarify and aligned the test cases accordingly.

**Hedge phrases** — the initial list was short. Researched common hedging patterns and expanded it significantly (`"it depends"`, `"may vary"`, `"generally speaking"`, `"to some extent"`, etc.).

---

## Feature 2 — Query Fan-Out Engine

### Build the optimizer before tuning

After getting basic fan-out generation working, the next questions were: what threshold marks a sub-query as "covered"? Is the prompt good enough? Rather than guessing, I built two small scripts first:

- `optimization/threshold_tuning/run_sweep.py` — sweeps cosine similarity thresholds against a hand-labelled dataset
- `optimization/prompt_tuning/run_prompt_eval.py` — evaluates prompt output across 5 fixed queries for structural compliance

This gave something to measure against instead of eyeballing outputs.

### Prompt iteration

#### Iteration 1 — First draft

The first prompt had no word-count constraint and included the line _"Follow the structure of the example exactly."_

Problem: sub-queries were too verbose.

```
# target query: "best AI writing tool for SEO"

use_case:
  - AI writing tools for SEO content creation for blogs   ← "X for Y for Z" chaining
  - best AI writing software for e-commerce SEO

trust_signals:
  - reviews of AI writing tools for SEO effectiveness
  - case studies on AI writing tools improving SEO rankings  ← sentence, not a search query
```

→ [iteration_1_too_long_subqueries.csv](../optimization/prompt_tuning/logs/iteration_1_too_long_subqueries.csv)

**What I changed:** Added a 4–9 word cap and "write like a real user" instruction. Removed the "follow the example exactly" line — that was causing the model to mirror the example's verbosity rather than enforcing type coverage.

---

#### Iteration 2 — Conciseness fixed, trust signals still wrong

Length was better. But `trust_signals` still read like academic research, not real searches:

```
trust_signals:
  - AI writing tool for SEO user reviews        ← generic, academic phrasing
  - case studies of AI writing for SEO          ← nobody types this into Google

# target query: "best CRM software for startups"
trust_signals:
  - CRM software startup user reviews
  - case studies of successful CRM implementations
```

→ [iteration_2_trust_signals_not_good.csv](../optimization/prompt_tuning/logs/iteration_2_trust_signals_not_good.csv)

**Root cause:** The example queries in the prompt used words like "reviews" and "case studies". The model latched onto that vocabulary and reproduced it across every domain. I researched real SERP patterns and Reddit threads on how people actually search for software recommendations — the vocabulary is more social: "used by", "popular among", "most-used at companies like...".

**What I changed:** Replaced the `trust_signals` example queries in the prompt:

```json
{"type": "trust_signals", "query": "project management tools used by large teams"},
{"type": "trust_signals", "query": "most popular PM software among engineering teams"},
```

Also updated the type description from _"reviews, testimonials, case studies"_ to _"social proof: used by teams/companies, most popular among professionals"_ — anchoring the model to adoption vocabulary instead of academic vocabulary.

---

#### Iteration 3 — Final prompt ✅

```
# target query: "best AI writing tool for SEO"
trust_signals:
  - most trusted AI writing tools for SEO
  - popular AI writing software among marketers

# target query: "best CRM software for startups"
trust_signals:
  - most popular CRM for startup sales teams
  - CRM tools trusted by early-stage companies
```

Output felt natural and appropriate across all 5 test queries.

→ [final_iteration_kept_results.csv](../optimization/prompt_tuning/logs/final_iteration_kept_results.csv)

---

#### Final prompt (as shipped)

This is the exact `_SYSTEM_PROMPT` constant that went into production in `app/services/fanout_engine.py`:

```
You are a query decomposition engine for an AI content optimisation platform.

YOUR ROLE
You simulate how AI search engines (Perplexity, ChatGPT Search, Google AI Mode) break
a user query into sub-queries before generating a comprehensive answer. Your output is
used to identify gaps in a content article.

TASK
Given a target query, generate between 10 and 15 sub-queries that span all six types
listed below. Cover the query topic thoroughly and generically — the same prompt must
work for any domain (SEO tools, CRM software, project management, etc.).

THE SIX SUB-QUERY TYPES
You must use EXACTLY these identifiers as the "type" field value:

  comparative      — how the subject compares against competitors or alternatives
  feature_specific — a specific feature, capability, or attribute
  use_case         — a concrete real-world application or audience segment
  trust_signals    — social proof: used by teams/companies, most popular among professionals, trusted by industry
  how_to           — a procedural, instructional, or "how do I" angle
  definitional     — a conceptual, explanatory, or "what is" angle

HARD CONSTRAINTS
1. Generate at least 2 sub-queries for EVERY type (12 minimum across all six types).
2. Total sub-queries must be between 10 and 15 inclusive.
3. Each sub-query must be a realistic search query string — not a sentence fragment.
4. Keep each query concise: 4–9 words maximum. No redundant prepositions or filler
   words (avoid patterns like "X for Y for Z" — collapse to "X for Y Z" instead).
5. Write queries the way a real user would type them into a search engine — natural,
   direct, no unnecessary repetition.
6. Return ONLY a valid JSON object. No markdown fences. No prose. No extra fields.
7. The JSON must match this schema exactly:
{
  "sub_queries": [
    {"type": "<one of the six types above>", "query": "<search query string>"},
    ...
  ]
}

EXAMPLE — for target query "best project management software for remote teams":
{
  "sub_queries": [
    {"type": "comparative", "query": "Asana vs Monday.com for remote teams"},
    {"type": "comparative", "query": "Jira vs ClickUp distributed engineering"},
    {"type": "feature_specific", "query": "PM tool with async video updates"},
    {"type": "feature_specific", "query": "project management time zone features"},
    {"type": "use_case", "query": "project management for remote agencies"},
    {"type": "use_case", "query": "best PM tool distributed startup teams"},
    {"type": "trust_signals", "query": "project management tools used by large teams"},
    {"type": "trust_signals", "query": "most popular PM software among engineering teams"},
    {"type": "how_to", "query": "how to manage remote projects without meetings"},
    {"type": "how_to", "query": "track distributed team progress in real time"},
    {"type": "definitional", "query": "what is asynchronous project management"},
    {"type": "definitional", "query": "remote-first project workflow definition"}
  ]
}

Now generate sub-queries for the target query provided by the user.
```

The user message is simply: `"Target query: {target_query}"`

---

#### Final prompt design rationale

- **System + user message split** — system message holds all the rules, schema, and example; user message is just `"Target query: {target_query}"`. Keeps model attention on the instructions.
- **`temperature=0.2`** — low enough to prevent hallucinated field names, high enough that phrasing varies across sub-queries instead of sounding repetitive.
- **JSON schema in the prompt** — without it, models wrap the array in different keys (`"queries"`, `"results"`, `"output"`) across calls. Showing the exact schema removes that ambiguity. The code also handles the envelope defensively in `parse_llm_response()`.

**Future improvement:** The hardcoded example is a project management query — for health or finance domains the model can lean too heavily on that phrasing. A small per-industry example bank with dynamic selection at request time would generalise better.

---

### Threshold tuning

Ran the sweep against a hand-labelled set of `(sub_query, content_chunk, label)` pairs:

```json
{
  "id": "s001",
  "sub_query": "how does Jasper AI compare to human writers for SEO content",
  "query_type": "comparative",
  "content_chunk": "We ran a head-to-head comparison between Jasper AI-generated articles and content written by professional SEO copywriters...",
  "label": true
}
```

Also compared `all-MiniLM-L6-v2` vs `all-mpnet-base-v2` on the same dataset — the accuracy drop with the smaller model was manageable, so kept MiniLM for the speed benefit.

The sweep picked `0.66` as the best F1 threshold. The assignment suggests `0.72` — lower recall at the same precision, meaning more false "gap" flags, which feels worse for a content tool.

→ [sweep results](../optimization/threshold_tuning/reports/sweep_20260410_194820.csv)

![Threshold sweep](threshold_optimization_coverage.png)

---

### Tests for GEO

Two kinds: offline mocked tests (error paths, retry exhaustion, malformed JSON, unknown types) and a live integration test that hits the real API for basic sanity. Kept them clearly separated.
