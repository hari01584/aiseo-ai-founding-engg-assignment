# AEGIS — Answer Engine & Generative Intelligence Suite

AI-powered content intelligence platform that scores and diagnoses content for **AEO** (Answer Engine Optimization) and **GEO** (Generative Engine Optimization).

Two live endpoints:

| Endpoint | Feature |
|---|---|
| `POST /api/aeo/analyze` | AEO Content Scorer — 3 NLP checks, score 0–100 |
| `POST /api/fanout/generate` | Query Fan-Out Engine — LLM sub-query generation + semantic gap analysis |

---

## 1. Installation & Running

### Prerequisites

- Python 3.11+
- An **OpenAI API key** (`gpt-4o-mini` is used by default)

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/hari01584/aiseo-ai-founding-engg-assignment.git
cd aiseo-ai-founding-engg-assignment

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy model (used by Check A)
python -m spacy download en_core_web_lg
# Fallback: en_core_web_sm is auto-tried if lg is unavailable

# 5. Set up your environment file
cp .env.sample .env
# Then open .env and paste your API key
```

`.env.sample` contains:

```env
# Paste your OpenAI API key here
OPENAI_API_KEY=sk-...
```

### Start the server

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## 2. Running Tests

```bash
pytest
```

The test suite is split into two top-level groups:

### `tests/aeo/` — AEO Scorer checks

| File | What it covers |
|---|---|
| [`tests/aeo/check_a_direct_answer_detection/test_first_paragraph.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_a_direct_answer_detection/test_first_paragraph.py) | First-paragraph extraction from HTML and plain text — edge cases like short paragraphs, nested tags, plain-text line breaks |
| [`tests/aeo/check_a_direct_answer_detection/test_count_words.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_a_direct_answer_detection/test_count_words.py) | Word-count scoring — boundary values at 60, 61, 90, 91 words |
| [`tests/aeo/check_a_direct_answer_detection/test_has_hedge_phrase.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_a_direct_answer_detection/test_has_hedge_phrase.py) | Hedge-phrase detection — all canonical phrases, case-insensitivity, phrases embedded mid-sentence |
| [`tests/aeo/check_a_direct_answer_detection/test_is_declarative.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_a_direct_answer_detection/test_is_declarative.py) | spaCy declarative check — confirms subject + root verb presence, rejects questions and fragments |
| [`tests/aeo/check_a_direct_answer_detection/test_check_a_direct_answer_full_integration.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_a_direct_answer_detection/test_check_a_direct_answer_full_integration.py) | Full `DirectAnswerCheck.run()` — every score band (0/8/12/20), plain text and HTML inputs, all output fields asserted |
| [`tests/aeo/check_b_htag_hierachy/test_htag_hierarchy.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_b_htag_hierachy/test_htag_hierarchy.py) | Full `HtagHierarchyCheck.run()` — perfect hierarchy, skipped levels, tag before H1, missing H1, 3+ violations; both raw soup and full HTML paths |
| [`tests/aeo/check_c_snippet_reader/test_sentence_splitting.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_c_snippet_reader/test_sentence_splitting.py) | `split_sentences` — handles abbreviations, trailing punctuation, short fragments |
| [`tests/aeo/check_c_snippet_reader/test_complexity_scoring.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_c_snippet_reader/test_complexity_scoring.py) | Per-sentence complexity (syllables ÷ words) — simple vs. technical sentences |
| [`tests/aeo/check_c_snippet_reader/test_top_complex_sentences.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_c_snippet_reader/test_top_complex_sentences.py) | `top_complex_sentences` ranking — correct top-3 ordering, truncation at 200 chars |
| [`tests/aeo/check_c_snippet_reader/test_check_b_snippet_reader_full_integration.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/check_c_snippet_reader/test_check_b_snippet_reader_full_integration.py) | Full `ReadabilityCheck.run()` — all FK score bands, boilerplate-stripping (nav/footer must NOT skew the score), plain text and HTML |
| [`tests/aeo/test_api_aeo.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/aeo/test_api_aeo.py) | End-to-end API for `POST /api/aeo/analyze` — URL input, HTML input, plain-text input; response envelope shape; score in [0,100]; valid band string |

### `tests/geo/` — Fan-Out Engine checks

| File | What it covers |
|---|---|
| [`tests/geo/test_query_fanout.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/geo/test_query_fanout.py) | `generate_sub_queries()` unit tests — all LLM calls mocked; happy path, fence-wrapped JSON, bad JSON on all retries → `LLMUnavailableError`, too few sub-queries, unknown types filtered |
| [`tests/geo/test_gap_analysis.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/geo/test_gap_analysis.py) | `gap_analyzer.py` — pure math tests (`l2_normalise`, `build_gap_summary`), full pipeline with real embeddings: related queries → covered, unrelated queries → not covered |
| [`tests/geo/test_api_fanout.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/tests/geo/test_api_fanout.py) | End-to-end API for `POST /api/fanout/generate` — no content (gap fields absent), with content (gap_summary present), LLM 503 error shape, gap analysis failure is non-fatal (returns 200), validation errors → 422 |

> **Tip — verify specific failure modes directly:**
> ```bash
> # LLM retry exhaustion → 503
> pytest tests/geo/test_query_fanout.py -k "all_retries_fail" -v
> pytest tests/geo/test_api_fanout.py -k "llm_failure" -v
>
> # Gap analysis crash → still returns 200 with sub-queries
> pytest tests/geo/test_api_fanout.py -k "gap_analysis_failure" -v
> ```

---

## 3. Optimization Scripts

These are standalone tuning scripts — not pytest tests. They were used during development to arrive at the final prompt and similarity threshold, and can be re-run at any time to reproduce the results.

### 3a. Threshold Tuning

Finds the optimal cosine similarity threshold for gap analysis using a hand-labelled dataset of `(sub_query, content_chunk, label)` triples.

```bash
python -m optimization.threshold_tuning.run_sweep
```

**How it works:** Sweeps thresholds from 0.40 → 0.95 in steps of 0.02, computing precision / recall / F1 at each value against [`optimization/data/similarity_samples.json`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/data/similarity_samples.json).

**Output:** [`optimization/threshold_tuning/reports/sweep_20260410_194820.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/threshold_tuning/reports/sweep_20260410_194820.csv)

**Chosen threshold: `0.66`** — F1 = 0.96, precision = 1.0, recall = 0.923. See Decision 4c below for the full reasoning.

### 3b. Prompt Tuning

Evaluates each prompt iteration across 5 fixed target queries and scores structural compliance (valid JSON, all 6 types present, ≥2 per type, count in range).

```bash
python -m optimization.prompt_tuning.run_prompt_eval
```

**Queries tested:** [`optimization/data/prompt_eval_queries.json`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/data/prompt_eval_queries.json)

**Iteration logs:**

| Log | What went wrong |
|---|---|
| [`iteration_1_too_long_subqueries.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/prompt_tuning/logs/iteration_1_too_long_subqueries.csv) | Sub-queries too verbose — "X for Y for Z" chaining |
| [`iteration_2_trust_signals_not_good.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/prompt_tuning/logs/iteration_2_trust_signals_not_good.csv) | `trust_signals` reads like academic queries ("case studies of...") not real user searches |
| [`final_iteration_kept_results.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/prompt_tuning/logs/final_iteration_kept_results.csv) | Final prompt — output felt most natural and appropriate across all 5 queries |

The full step-by-step thinking, observations, and changes for each iteration are in [`my_journey/how_i_made_records.md`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/my_journey/how_i_made_records.md).

---

## 4. Key Engineering Decisions

### 4a. LLM JSON Reliability

LLMs don't always return clean JSON even when asked explicitly. Three layers of defence are in [`app/services/fanout_engine.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/app/services/fanout_engine.py):

1. **Fence stripping** — `strip_markdown_fences()` removes ` ```json ... ``` ` wrappers that models sometimes add despite being told not to.
2. **Pydantic validation** — each raw dict is passed through the `SubQuery` model. Items with unknown `type` values are silently dropped rather than crashing — the list is re-checked for minimum count constraints after filtering.
3. **Retry with exponential back-off** — up to 3 attempts, delay doubling each time (`1s → 2s → 4s`). After all retries fail, `LLMUnavailableError` is raised and the API returns a clean `503` with the error detail string.

Verify the retry and 503 path:
```bash
pytest tests/geo/test_query_fanout.py -k "all_retries_fail" -v
pytest tests/geo/test_api_fanout.py -k "llm_failure" -v
```

### 4b. Embedding Model Choice

`all-MiniLM-L6-v2` was chosen over `all-mpnet-base-v2`:

- ~5× faster on CPU (~22 ms vs ~110 ms per sentence)
- 384-dim vs 768-dim — lower memory footprint, matters at scale
- Running the threshold sweep with both models showed the best-F1 threshold stabilised at the same value (`0.66`), confirming MiniLM is accurate enough for sentence-level semantic similarity at this task

For a GPU-backed production service, `mpnet` would be worth reconsidering for harder comparative queries. For this use case the speed win is clear and the accuracy difference is negligible on the labelled set.

### 4c. Similarity Threshold

The assignment suggests `0.72`. After running the threshold sweep, **`0.66`** was chosen instead.

Full results: [`optimization/threshold_tuning/reports/sweep_20260410_194820.csv`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/optimization/threshold_tuning/reports/sweep_20260410_194820.csv)

| Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 0.66 | 1.00 | 0.923 | **0.96** ✅ chosen |
| 0.68 | 1.00 | 0.846 | 0.917 |
| 0.72 | 1.00 | 0.846 | 0.917 |
| 0.80 | 1.00 | 0.692 | 0.818 |

`0.72` achieves F1 = 0.917 — not bad, but `0.66` reaches F1 = 0.96 with equal precision and meaningfully better recall. In a content tool, false "gap" flags (telling users their content doesn't cover something it actually does) are more damaging to trust than the reverse, so the higher-recall threshold is the right trade-off here. The threshold is a single constant (`DEFAULT_SIMILARITY_THRESHOLD` in `gap_analyzer.py`) — trivial to update as more labelled data arrives.

### 4d. Content Parsing Robustness

Handled in [`app/services/content_parser.py`](https://github.com/hari01584/aiseo-ai-founding-engg-assignment/blob/main/app/services/content_parser.py):

- **Boilerplate stripping** — `nav`, `header`, `footer`, `aside`, `script`, `style`, `noscript` are removed before computing the readability score and extracting body text. This prevents a site's navigation menu from skewing the FK grade level.
- **Plain-text fallback** — if the input does not look like HTML, the parser takes a plain-text path splitting on double newlines. The API works equally well for pasted raw text.
- **URL fetch** — `httpx` with a 10s timeout and a bot `User-Agent`. Timeout, HTTP errors, and network failures all map to a `ValueError` → `422 Unprocessable Entity` with the original error message.

**Known limitation (intentional simplification):** First-paragraph extraction uses the first `<p>` tag. Real sites are messier — see "What I'd Improve" below.

### 4e. Failure Modes

| Failure | Behaviour |
|---|---|
| URL unreachable / timeout | `422` with `error: url_fetch_failed` and the exception message |
| LLM returns bad JSON on all retries | `503` with `error: llm_unavailable` and `JSONDecodeError` detail |
| Gap analysis crashes (e.g. embedding model error) | `200` — sub-queries still returned; `gap_summary` is `null` (non-fatal degradation) |
| Blank `input_value` or `target_query` | `422` from Pydantic field validation before any service code runs |

```bash
# Verify gap analysis failure is non-fatal
pytest tests/geo/test_api_fanout.py -k "gap_analysis_failure" -v
```

---

## 5. Scope & What Was Skipped

The solution covers all required deliverables:

- ✅ `POST /api/aeo/analyze` — all 3 checks with correct scoring formula and bands
- ✅ `POST /api/fanout/generate` — LLM fan-out with all 6 sub-query types, retry logic, Pydantic validation
- ✅ Semantic gap analysis — `sentence-transformers`, L2-normalised cosine similarity, threshold-based coverage labelling
- ✅ Unit + integration tests for every AEO check function, plus API-level and mocked-LLM tests for GEO
- ✅ Optimization scripts (threshold sweep + prompt eval) with logged artifacts

**Focus distribution:** Most deliberate engineering effort went into Feature 2 — specifically the prompt iteration process and the threshold tuning methodology — because those are where the AI judgement calls are hardest to get right and easiest to do poorly.

**Content parsing is intentionally simplified.** First-paragraph extraction uses the first `<p>` tag. This is sufficient for well-structured articles but breaks on real-world sites in several ways:

- **JavaScript-rendered pages** — `httpx` fetches static HTML only. SPAs return an empty or near-empty body. The correct fix is a headless browser (Playwright or Selenium) for URL inputs.
- **Complex layouts** — many real pages have short `<p>` tags inside cookie banners, ad containers, or pull-quote divs that appear before the article body. A robust implementation would skip paragraphs below a minimum meaningful length (e.g. < 15 words), prefer `<article>` or `<main>` blocks, and fall back to the second/third paragraph if the first looks like UI chrome.
- **Login-walled pages** — the fetcher has no session/cookie support; these return a login form rather than article content.

These are real production concerns but out of scope for a 6–8 hour assignment. The architecture (a single `fetch_and_parse` function with a clearly defined interface) makes them straightforward to address later.

---

## 6. What I'd Improve With More Time

**Feature 1 — AEO Scorer**

- Replace the first-`<p>` heuristic with a proper article-body extractor (e.g. `trafilatura`, or a `<main>`/`<article>` tag search with minimum word-count filtering) so the parser works correctly on real news and marketing pages.
- Add Playwright/Selenium support for JS-rendered URLs — even a simple `"js_render": true` flag in the request body would handle the most common case.
- Make the hedge phrase list configurable from a file rather than hardcoded in source, so it can be extended without a code change.

**Feature 2 — Query Fan-Out Engine**

- The current prompt uses static hardcoded examples. A production system would maintain a small per-industry example bank (SaaS, e-commerce, healthcare, etc.) and inject the closest-matching example at request time — reducing the chance the model overfits to the sample domain.
- Sub-query generation could be pipelined: one prompt per type (6 parallel async calls), each with a dedicated critic pass that rewrites any query that reads like a sentence fragment rather than a real search. Higher quality at the cost of more latency — acceptable with async.
- Expand the hand-labelled similarity dataset beyond 20 samples (aim for 100+, balanced across industries and difficulty levels) for a more statistically reliable F1 estimate and threshold selection.

**Infrastructure**

- Cache the embedding model in a shared memory object between workers (currently it is loaded fresh per process).
- Add a `GET /health` endpoint reporting spaCy and sentence-transformer load status.
- Rate-limit the URL fetch path to prevent abuse.
