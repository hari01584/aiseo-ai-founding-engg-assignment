# Optimization Scripts

Two standalone runner scripts for tuning the gap analysis threshold and the fan-out LLM prompt. Neither script is a `pytest` test — they are run directly and produce artifact files for human review.

---

## Threshold Tuning

**Purpose:** Find the optimal cosine similarity threshold for the gap analyzer using hand-labelled data.

**Dataset:** `data/similarity_samples.json` — 20 labelled `(sub_query, content_chunk, label)` triples.  
Add more rows to improve reliability. The `label` field is your human judgement: does this content chunk cover this sub-query?

**Run:**
```bash
python -m optimization.threshold_tuning.run_sweep
```

**Output:** `threshold_tuning/reports/sweep_<timestamp>.csv`

Each row is one threshold value with precision / recall / F1 / confusion matrix counts.  
The script also prints a table to stdout with the best F1 row marked.

**Workflow:**
1. Add samples to `data/similarity_samples.json` (aim for 50+, balanced across query types and difficulty levels)
2. Run the sweep
3. Read the CSV — pick the threshold with the best F1 for your use case (or tune precision/recall trade-off manually)
4. Update `DEFAULT_SIMILARITY_THRESHOLD` in `app/services/gap_analyzer.py`

---

## Prompt Tuning

**Purpose:** Compare multiple prompt versions on structural compliance (JSON validity, type coverage, count constraints) across a fixed set of queries.

**Dataset:** `data/prompt_eval_queries.json` — 5 fixed target queries used for reproducibility across runs.

**Run:**
```bash
python -m optimization.prompt_tuning.run_prompt_eval
```

**Output:** `prompt_tuning/logs/run_<timestamp>.csv`

Each row is one `(prompt_version × query)` call with:
- All structural scores (json_valid, total_in_range, all_6_types_present, min_2_per_type, no_extra_top_level_keys)
- `composite_score` (0.0–1.0)
- `scorer_notes` — human-readable explanation of any failures
- `raw_llm_response` — the exact string the LLM returned

**Workflow:**
1. Run the eval — reads all prompt versions from `app/services/fanout_prompts.py`
2. Open the CSV in a spreadsheet
3. Read `raw_llm_response` for failing rows
4. Edit the prompt in `app/services/fanout_prompts.py` (add a new version, e.g. `PROMPT_V4`)
5. Run again — a new timestamped CSV is created so you keep full history
6. Compare composite scores across runs to confirm improvement

---

## Data files

| File | Purpose |
|---|---|
| `data/similarity_samples.json` | Human-labelled pairs for threshold sweep |
| `data/prompt_eval_queries.json` | Fixed queries for prompt evaluation |

Add to `similarity_samples.json` freely — more data = better threshold estimate.  
Do not change `prompt_eval_queries.json` between runs (it would break comparability).
