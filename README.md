# 🧪 Take-Home Assignment — AI Engineer
## AEGIS — Answer Engine & Generative Intelligence Suite

> **Time Budget:** 6–8 hours  
> **Submission Deadline:** 72 hours from receipt  
> **Role:** AI Engineer (Mid–Senior)

---

## 👋 Welcome

Thank you for making it to this stage of the interview process.

This assignment reflects the real work you'd be doing here — designing LLM pipelines, engineering prompts, building NLP scoring systems, and making thoughtful decisions about when to use an LLM and when a simpler method is better.

We evaluate **your reasoning as much as your output**. A focused solution that covers 60% of the scope with excellent prompt design and clean NLP code beats a rushed implementation that covers everything poorly.

**Read this document fully before writing a single line of code.**

---

## 🧭 Background — What is AEGIS?

The way people discover information is changing rapidly. Instead of ten blue links, users now receive direct answers from AI systems: **ChatGPT Search**, **Perplexity**, **Google AI Mode**, and **Gemini**.

This creates a new challenge for content teams: *How do you ensure your content gets cited, surfaced, and recommended by these AI systems rather than a competitor's?*

Two disciplines have emerged:

- **AEO (Answer Engine Optimization)** — Structuring content so AI search engines can extract clean, direct answers (think: featured snippets, AI overviews, passage indexing).
- **GEO (Generative Engine Optimization)** — Making content citation-worthy and factually dense enough that LLMs actively choose to cite it when generating answers.

**AEGIS** is a content intelligence platform that scores, diagnoses, and improves content for AEO and GEO — and monitors how brands are represented across AI platforms in near-real-time.

---

## 🗺️ System Overview

AEGIS has four core modules. You will build parts of **two of them**. Understanding the full picture will help you make better architectural decisions.

```
┌───────────────────────────────────────────────────────────────┐
│                        AEGIS Platform                          │
│                                                                │
│  ┌─────────────────┐  ┌─────────────────┐  ┌───────────────┐ │
│  │  AEO Optimizer  │  │  GEO Optimizer  │  │ Brand Monitor │ │
│  │                 │  │                 │  │               │ │
│  │ NLP checks for  │  │ LLM + embedding │  │ Track brand   │ │
│  │ AI answer       │  │ checks for      │  │ mentions on   │ │
│  │ extraction      │  │ citation-worth  │  │ AI platforms  │ │
│  └─────────────────┘  └─────────────────┘  └───────────────┘ │
│                                                                │
│  ┌────────────────────────────────────────────────────────┐   │
│  │                  Query Fan-Out Engine                   │   │
│  │  LLM simulates how AI search decomposes a query →      │   │
│  │  embeddings detect content coverage gaps               │   │
│  └────────────────────────────────────────────────────────┘   │
└───────────────────────────────────────────────────────────────┘
```

---

## 🎯 Your Assignment

### What To Build

You will build a **Python API service** implementing two AI-powered features:

| Feature | Core AI Challenge |
|---------|------------------|
| **Feature 1 — AEO Content Scorer** | NLP pipeline to score content across 3 structured checks |
| **Feature 2 — Query Fan-Out Engine** | LLM prompt engineering + semantic similarity gap analysis |

---

## Feature 1 — AEO Content Scorer

A REST endpoint that accepts a URL or pasted content, applies **three NLP-based checks**, and returns an **AEO Readiness Score (0–100)** with per-check diagnostics and recommendations.

The core engineering challenge here is building a **clean, modular NLP scoring pipeline** — each check should be independently testable and the system should be easy to extend with additional checks.

---

### Check A — Direct Answer Detection

**What it tests:** Does the first paragraph answer the likely primary query in 60 words or fewer, with a clear declarative statement?

**Implementation requirements:**

1. Extract the first paragraph from the content (first `<p>` tag in HTML, or first paragraph break in plain text)
2. Word count: must be ≤ 60 to pass
3. Sentence completeness check: the paragraph should be a declarative statement — not a question, not a fragment. Use spaCy's dependency parser to verify subject + root verb presence
4. Hedge phrase detection: penalize if any of these appear — `"it depends"`, `"may vary"`, `"in some cases"`, `"this varies"`, `"generally speaking"`

**Scoring (max 20 pts):**

| Condition | Score |
|-----------|-------|
| ≤ 60 words + declarative + no hedge | 20 |
| ≤ 60 words but hedging or incomplete | 12 |
| 61–90 words | 8 |
| > 90 words | 0 |

**Expected output fields:** `word_count`, `has_hedge_phrase`, `is_declarative`, `recommendation`

---

### Check B — H-tag Hierarchy Checker

**What it tests:** Is there a valid, logical heading structure (H1 → H2 → H3)?

**Implementation requirements:**

1. Parse all H-tags in DOM order using BeautifulSoup
2. Validate the following rules:
   - Exactly one `<h1>` is present
   - No heading level is skipped (H1 → H3 without an H2 is a violation)
   - No H-tag appears before the H1

**Scoring (max 20 pts):**

| Condition | Score |
|-----------|-------|
| 0 violations | 20 |
| 1–2 violations | 12 |
| 3+ violations OR missing H1 | 0 |

**Expected output fields:** `violations` (list of rule violations with description), `h_tags_found` (ordered list), `recommendation`

---

### Check C — Snippet Readability Scorer

**What it tests:** Is the content readable at Flesch-Kincaid Grade Level 7–9? This is the target range for AI answer extraction — complex enough to be credible, simple enough to be extractable.

**Implementation requirements:**

1. Strip boilerplate (nav, footer, header) from HTML before scoring
2. Compute Flesch-Kincaid Grade Level using `textstat`
3. Identify the **3 most complex sentences** in the text (rank by: syllable count ÷ word count per sentence)

**Scoring (max 20 pts):**

| FK Grade Level | Score |
|----------------|-------|
| 7–9 | 20 |
| 6 or 10 | 14 |
| 5 or 11 | 8 |
| ≤ 4 or ≥ 12 | 0 |

**Expected output fields:** `fk_grade_level`, `target_range`, `complex_sentences` (top 3), `recommendation`

---

### Score Aggregation

```
Raw Score = Check A + Check B + Check C   (max 60 pts)
AEO Readiness Score = (Raw Score / 60) × 100   (normalized to 100)
```

**Score Bands:**

| Range | Label |
|-------|-------|
| 85–100 | AEO Optimized ✅ |
| 65–84 | Needs Improvement 🟡 |
| 40–64 | Significant Gaps 🔴 |
| 0–39 | Not AEO Ready ⛔ |

---

## Feature 2 — Query Fan-Out Engine

This is the more open-ended, AI-heavy feature. It tests your ability to **design LLM prompts for structured output**, handle model responses defensively, and apply **semantic embeddings** for content gap analysis.

### What it does

1. User provides a target query: e.g., `"best AI writing tool for SEO"`
2. Your service calls an LLM to generate **10–15 sub-queries** across 6 query types — simulating how AI search engines decompose a query to build a comprehensive answer
3. Each sub-query is checked against the user's provided content using **sentence-transformer embeddings + cosine similarity**
4. Returns a gap map: which sub-query types are covered, which are missing

---

### The 6 Sub-Query Types

| Type | Description | Example |
|------|-------------|---------|
| `comparative` | Query vs. alternatives | "AI writing tool vs human writers for SEO" |
| `feature_specific` | Specific capability focus | "AI writing tool with keyword clustering" |
| `use_case` | Real-world application | "AI writing tool for agency content at scale" |
| `trust_signals` | Reviews, credibility, proof | "AI SEO writing tool case studies 2025" |
| `how_to` | Procedural / instructional | "how to use AI to optimize blog content for SEO" |
| `definitional` | Conceptual / "what is" | "what is AI-assisted SEO content writing" |

---

### Prompt Engineering Requirements

Your LLM prompt is a **core deliverable** — we will read it carefully. It must:

- Clearly specify the 6 sub-query types and require at least 2 from each
- Instruct the model to return a **valid JSON object** (not markdown, not prose)
- Be robust to the query topic — it should work equally well for `"best CRM software"` as for `"best AI writing tool for SEO"`
- Include a concrete example of the expected output format in the prompt itself

You may use any LLM.

> 💡 Think carefully about: How do you prevent the model from hallucinating extra fields? How do you handle the case where the model returns fewer than 10 sub-queries? How do you validate the JSON before returning it to the caller?

---

### Semantic Gap Analysis (Required)

Once sub-queries are generated, check content coverage using embeddings:

1. Vectorize the user's content into sentence-level chunks using `sentence-transformers` (`all-mpnet-base-v2` or `all-MiniLM-L6-v2`)
2. Vectorize each sub-query
3. For each sub-query, compute the **max cosine similarity** against all content chunks
4. If max similarity ≥ **0.72** → mark sub-query as `covered: true`
5. If max similarity < 0.72 → mark as `covered: false` (gap)

> **Why 0.72?** This threshold is a starting point. In your README, explain whether you'd tune this, and how.

---

## 📦 Deliverables

### 1. Working API (Required)
- Language: **Python**
- Framework: **FastAPI**
- Two working endpoints:

```
POST /api/aeo/analyze
POST /api/fanout/generate
```

### 2. Your README (Required)
Your `README.md` must cover:
- Exact commands to run the project locally
- All required environment variables (API keys etc.)
- Which parts you completed vs. skipped, and why
- Your prompt design decisions — what you tried, what worked, what didn't
- The gap analysis threshold: why 0.72, or what value you chose and why
- What you'd improve with more time

### 3. Tests (Required)
- Unit tests for **all three AEO check functions** (not just the endpoint)
- At minimum: one passing case and one failing case per check
- Framework: `pytest`
- Bonus: a test that mocks the LLM call and validates your JSON parsing logic

### 4. (Bonus) Prompt Iteration Log
A `PROMPT_LOG.md` file documenting:
- Your first prompt draft
- What was wrong with it (hallucinations? wrong format? missing types?)
- What you changed and why
- Your final prompt

This is one of the best signals of an experienced AI engineer. Even a brief 1-page log is valuable.

---

## 📡 API Contracts

### `POST /api/aeo/analyze`

**Request:**
```json
{
  "input_type": "url",
  "input_value": "https://example.com/article"
}
```
```json
{
  "input_type": "text",
  "input_value": "Your raw HTML or plain text content..."
}
```

**Response `200 OK`:**
```json
{
  "aeo_score": 72,
  "band": "Needs Improvement",
  "checks": [
    {
      "check_id": "direct_answer",
      "name": "Direct Answer Detection",
      "passed": false,
      "score": 8,
      "max_score": 20,
      "details": {
        "word_count": 74,
        "threshold": 60,
        "is_declarative": true,
        "has_hedge_phrase": false
      },
      "recommendation": "Your opening paragraph is 74 words. Trim it to under 60 words with a direct, declarative answer."
    },
    {
      "check_id": "htag_hierarchy",
      "name": "H-tag Hierarchy",
      "passed": true,
      "score": 20,
      "max_score": 20,
      "details": {
        "violations": [],
        "h_tags_found": ["h1", "h2", "h2", "h3", "h2"]
      },
      "recommendation": null
    },
    {
      "check_id": "readability",
      "name": "Snippet Readability",
      "passed": false,
      "score": 8,
      "max_score": 20,
      "details": {
        "fk_grade_level": 11.2,
        "target_range": "7-9",
        "complex_sentences": [
          "The multifaceted nature of contemporary content optimization...",
          "Heuristically speaking, syntactic structures that...",
          "Upon examination of longitudinal data sets..."
        ]
      },
      "recommendation": "Content reads at Grade 11.2. Shorten sentences and replace technical jargon with plain language to reach Grade 7–9."
    }
  ]
}
```

**Error `422 Unprocessable Entity`** (invalid URL, unreachable page, etc.):
```json
{
  "error": "url_fetch_failed",
  "message": "Could not retrieve content from the provided URL.",
  "detail": "Connection timeout after 10s"
}
```

---

### `POST /api/fanout/generate`

**Request:**
```json
{
  "target_query": "best AI writing tool for SEO",
  "existing_content": "Optional — paste article text here to enable gap analysis"
}
```

**Response `200 OK`:**
```json
{
  "target_query": "best AI writing tool for SEO",
  "model_used": "gemini-1.5-flash",
  "total_sub_queries": 13,
  "sub_queries": [
    {
      "type": "comparative",
      "query": "Jasper AI vs Surfer SEO vs Clearscope for content optimization",
      "covered": false,
      "similarity_score": 0.41
    },
    {
      "type": "feature_specific",
      "query": "AI writing tool with real-time SERP analysis",
      "covered": true,
      "similarity_score": 0.79
    },
    {
      "type": "trust_signals",
      "query": "AI SEO content tool reviews from marketing agencies 2025",
      "covered": false,
      "similarity_score": 0.38
    }
  ],
  "gap_summary": {
    "covered": 5,
    "total": 13,
    "coverage_percent": 38,
    "covered_types": ["feature_specific", "use_case"],
    "missing_types": ["comparative", "trust_signals", "how_to", "definitional"]
  }
}
```

> **Note:** `covered`, `similarity_score`, and `gap_summary` are only present when `existing_content` is provided. If no content is provided, omit these fields and return sub-queries only.

**Error — LLM failure `503 Service Unavailable`:**
```json
{
  "error": "llm_unavailable",
  "message": "Fan-out generation failed. The LLM returned an invalid response after 3 retries.",
  "detail": "JSONDecodeError on attempt 3"
}
```

---

## 🛠️ Technical Guidelines

### Stack
- **Language:** Python 3.11+
- **Framework:** FastAPI + Uvicorn
- **NLP:** spaCy (`en_core_web_sm` minimum, `en_core_web_lg` preferred), `textstat`
- **Embeddings:** `sentence-transformers` (`all-MiniLM-L6-v2` for speed, `all-mpnet-base-v2` for accuracy — your call, justify it)
- **LLM:** Gemini 1.5 Flash or GPT-4o-mini
- **HTML Parsing:** BeautifulSoup4 + `httpx` for URL fetching

### Suggested `requirements.txt`
```
fastapi
uvicorn[standard]
httpx
beautifulsoup4
spacy
textstat
sentence-transformers
torch                    # required by sentence-transformers
google-generativeai      # if using Gemini
openai                   # if using OpenAI
pydantic>=2.0
pytest
pytest-asyncio
```

### Key Engineering Decisions to Address in Your README

**1. LLM JSON reliability**  
LLMs don't always return valid JSON. How do you handle a malformed response? Do you retry? Use a fallback? Validate schema with Pydantic before returning?

**2. Embedding model choice**  
`all-MiniLM-L6-v2` is 5× faster than `all-mpnet-base-v2` but less accuracy. Which did you pick and why? For a production system, what would you consider?

**3. Similarity threshold**  
0.72 is provided as a starting point. Did you keep it? Test it? Tune it? Explain your reasoning.

**4. Content parsing robustness**  
How do you handle pages with no clear first paragraph? JavaScript-heavy pages that return empty HTML? Pages behind a login wall?

**5. Failure modes**  
What happens if the LLM times out mid-request? Does your API return something useful or does it throw a 500?

---

## ✅ Evaluation Criteria

We will score your submission across these dimensions:

| Dimension | Weight | What We Look For |
|-----------|--------|-----------------|
| **Prompt Engineering Quality** | 25% | Is the prompt well-structured, defensive, and clearly designed? Does it produce consistent, valid JSON? |
| **NLP Pipeline Design** | 25% | Are checks modular, independently testable, and correct on standard cases? |
| **AI System Thinking** | 20% | Do your README answers show you've thought about failure modes, model tradeoffs, and threshold tuning? |
| **Code Quality** | 15% | Readable, typed, consistent. Pydantic models used correctly. |
| **Testing** | 15% | Do tests cover real edge cases and the LLM error path? |

### Green Flags 🟢
- A prompt with explicit JSON schema in the system message or examples
- Retry logic on LLM calls with exponential backoff
- spaCy used thoughtfully (not just `nlp(text)` with no explanation)
- Honest discussion in your README about where your implementation falls short
- A `PROMPT_LOG.md` showing your iteration process
- Choosing `all-MiniLM-L6-v2` over `all-mpnet-base-v2` and justifying it with latency reasoning

### Red Flags 🔴
- Prompt that's a single sentence with no structure
- No handling when the LLM returns invalid JSON — API crashes
- Cosine similarity computed incorrectly (using raw dot product on non-normalized vectors)
- No tests, or tests that mock everything and verify nothing real
- A README that says "install requirements, run the app" and nothing else

---

## 📤 Submission Instructions

1. Push your code to a **private GitHub repository**
2. Add `[RECRUITER_GITHUB_HANDLE]` as a collaborator
3. Email the repo link to `[RECRUITER_EMAIL]` with subject:  
   `AEGIS AI Engineer Assignment — [Your Name]`
4. In the email body, include:
   - Time spent
   - The one decision you're least confident about
   - The one thing you'd improve with another 2 hours

---

## ❓ FAQ

**Can I use AI assistants while working on this?**  
Yes — use whatever you normally use. We will discuss your solution in the follow-up interview and expect you to explain every decision clearly. The prompt engineering section is especially important to own.

**What if I can't complete everything?**  
Ship what you can. A clean Feature 1 with excellent tests beats two broken features. Tell us clearly in your README what you skipped and why.

**My LLM call is slow / rate-limited. What should I do?**  
Build in retry logic. If you're hitting free-tier limits, cache a fixed LLM response for testing purposes — just document that you did this.

**Can I change the API contract?**  
Reasonably, yes. Document your changes and the reason in your README.

**Should I use async or sync for the FastAPI endpoints?**  
Your call — but sentence-transformers and spaCy are CPU-bound, and LLM calls are I/O-bound. Think about what concurrency model makes sense here and mention it in your README.

---

## 📁 Suggested Project Structure

```
aegis-assignment/
├── ASSIGNMENT_README.md       # This file
├── README.md                  # Your README
├── PROMPT_LOG.md              # (Bonus) Prompt iteration log
├── requirements.txt
├── app/
│   ├── main.py                # FastAPI app
│   ├── api/
│   │   ├── aeo.py             # AEO router + endpoint
│   │   └── fanout.py          # Fan-out router + endpoint
│   ├── services/
│   │   ├── content_parser.py  # HTML fetch + parse + boilerplate strip
│   │   ├── aeo_checks/
│   │   │   ├── base.py        # BaseCheck abstract class
│   │   │   ├── direct_answer.py
│   │   │   ├── htag_hierarchy.py
│   │   │   └── readability.py
│   │   ├── fanout_engine.py   # LLM call, prompt, response parsing
│   │   └── gap_analyzer.py    # Embedding + cosine similarity logic
│   └── models/
│       └── schemas.py         # All Pydantic request/response models
└── tests/
    ├── test_direct_answer.py
    ├── test_htag_hierarchy.py
    ├── test_readability.py
    └── test_fanout_parsing.py  # Tests LLM response JSON parsing (mocked)
```

---

Good luck — we're genuinely excited to see how you approach this. 🚀