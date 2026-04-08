# LLM Response Quality Evaluator

![System Architecture](LLM_Voice_Eval_Workflow.png)

Automated multi-dimensional evaluation of voice AI responses using the **LLM-as-Judge** pattern. Built for BloomingHealth's voice AI platform to replace manual call quality review.

Evaluations are scored across 6 healthcare-relevant dimensions, persisted to SQLite for trend analysis, and aggregatable by agent, prompt version, or call purpose.

---

## Setup

**Option A — uv (recommended, fastest):** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# 1. Install uv (skip if already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Configure environment
cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY

# 4. Start the server
uv run python main.py
# → API live at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs
```

**Option B — pip:**

```bash
pip install anthropic fastapi uvicorn python-dotenv pydantic httpx langdetect

cp .env.example .env
# Open .env and add your ANTHROPIC_API_KEY

python main.py
```

**`.env` options:**
```bash
ANTHROPIC_API_KEY=sk-ant-...

# Model selection — pick one:
JUDGE_MODEL=claude-haiku-4-5-20251001   # development: fast (~4s), cheap (~$0.50/1k evals)
JUDGE_MODEL=claude-sonnet-4-6           # production: best discrimination gap (default)
JUDGE_MODEL=claude-opus-4-6             # clinical: perfect calibration (std_dev=0.00)
```

**SQLite:** `evaluations.db` is created automatically in the project directory on the first `/api/evaluate` call. No configuration or migration needed. It is gitignored.

---

## Run Tests

```bash
# uv
uv sync --extra dev
uv run pytest tests/ -v

# pip
pip install pytest pytest-asyncio
pytest tests/ -v
```

14 tests covering: evaluation, caching, batch processing, A/B comparison, improvement, edge cases (empty, long, non-English), SQLite persistence, and pattern analysis.

---

## API Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/evaluate` | POST | Score a single response across 6 dimensions |
| `/api/evaluate/batch` | POST | Score multiple responses + aggregate statistics |
| `/api/compare` | POST | A/B comparison of two responses to the same context |
| `/api/improve` | POST | Generate an improved version and score both |
| `/api/evaluate/calibrate` | POST | Measure scoring consistency across N independent runs |
| `/api/analysis/patterns` | GET | Aggregated score trends by agent, prompt version, or call purpose |
| `/health` | GET | Health check |

---

### `POST /api/evaluate`

Scores a single response. Pass optional `metadata` to tag evaluations for pattern analysis.

```bash
curl -X POST http://localhost:8000/api/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "context": {
      "current_directive": "Ask about food security using USDA screening questions",
      "user_input": "We sometimes run out of food before the end of the month"
    },
    "response": "I understand that can be really challenging. Would you say that happens often, sometimes, or rarely?",
    "metadata": {
      "agent_id": "food_security_agent",
      "prompt_version": "v2.1",
      "call_purpose": "sdoh_screening"
    }
  }'
```

**Response:**
```json
{
  "overall_score": 8.5,
  "dimensions": {
    "task_completion": { "score": 9, "reasoning": "Successfully asked clarifying question" },
    "empathy":         { "score": 8, "reasoning": "Acknowledged difficulty appropriately" },
    "conciseness":     { "score": 9, "reasoning": "Brief and focused" },
    "naturalness":     { "score": 8, "reasoning": "Conversational tone for voice" },
    "safety":          { "score": 10, "reasoning": "No harmful content" },
    "clarity":         { "score": 8, "reasoning": "Clear when spoken aloud" }
  },
  "flags": [],
  "suggestions": ["Consider warmer acknowledgment before the clarifying question"],
  "cached": false
}
```

**Automatic flags (detected before the LLM call):**

| Flag | Trigger |
|---|---|
| `empty_response` | Response is blank |
| `response_too_long_for_voice` | Response exceeds 500 characters (~30s to speak) |
| `non_english_response:<lang>` | Non-English content detected (e.g. `non_english_response:es`) |

---

### `POST /api/evaluate/batch`

Evaluates multiple responses in one call and returns individual scores plus aggregate statistics.

```bash
curl -X POST http://localhost:8000/api/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "evaluations": [
      {
        "context": { "current_directive": "Verify date of birth", "user_input": "March 15th, 1985" },
        "response": "Got it, March 15th, 1985. Thank you for confirming.",
        "metadata": { "agent_id": "intake_agent", "prompt_version": "v1" }
      },
      {
        "context": { "current_directive": "Verify date of birth", "user_input": "March 15th, 1985" },
        "response": "Perfect! I have recorded your date of birth as March 15th, 1985. Is there anything else I can help you with today?",
        "metadata": { "agent_id": "intake_agent", "prompt_version": "v2" }
      }
    ]
  }'
```

**Response:**
```json
{
  "results": [ { "overall_score": 8.3, ... }, { "overall_score": 7.7, ... } ],
  "aggregate": {
    "count": 2,
    "mean_overall": 8.0,
    "min_overall": 7.7,
    "max_overall": 8.3,
    "dimension_means": {
      "task_completion": 8.5,
      "empathy": 7.0,
      "conciseness": 8.0,
      "naturalness": 8.5,
      "safety": 9.5,
      "clarity": 8.5
    }
  }
}
```

---

### `POST /api/compare`

A/B comparison of two responses to the same context. Uses a single LLM call for consistent relative judgment.

```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "context": {
      "current_directive": "Verify date of birth",
      "user_input": "March 15th, 1985"
    },
    "response_a": "Got it, March 15th, 1985. Thank you.",
    "response_b": "Perfect! I have recorded your date of birth. Is there anything else I can help you with today?"
  }'
```

**Response:**
```json
{
  "winner": "a",
  "comparison": {
    "conciseness":     { "winner": "a", "reasoning": "A is appropriately brief; B adds unnecessary filler" },
    "task_completion": { "winner": "tie", "reasoning": "Both confirm the DOB accurately" },
    "naturalness":     { "winner": "a", "reasoning": "A sounds more natural for voice" },
    "empathy":         { "winner": "tie", "reasoning": "Neither requires emotional attunement here" },
    "safety":          { "winner": "tie", "reasoning": "Both are safe" },
    "clarity":         { "winner": "a", "reasoning": "A is clearer when spoken aloud" }
  },
  "recommendation": "Prefer Response A for concise verification — avoid filler phrases in voice contexts."
}
```

---

### `POST /api/improve`

Identifies weak dimensions (score < 8) and rewrites the response targeting those specifically.

```bash
curl -X POST http://localhost:8000/api/improve \
  -H "Content-Type: application/json" \
  -d '{
    "context": {
      "current_directive": "Handle user confusion about survey purpose",
      "user_input": "Why are you asking me all these personal questions?"
    },
    "response": "I am an AI assistant. These questions are part of a standard SDOH assessment protocol."
  }'
```

**Response:**
```json
{
  "original_score": 5.2,
  "improved_response": "That is a fair question. This survey helps us understand if there are any local resources — like food assistance or transportation — that could make a difference for you. Everything you share stays private.",
  "improved_score": 8.7,
  "changes_made": [
    "Validated the user's concern with 'That is a fair question'",
    "Replaced clinical jargon (SDOH protocol) with plain-language explanation",
    "Added confidentiality assurance to reduce hesitation"
  ]
}
```

---

### `POST /api/evaluate/calibrate`

Runs the same evaluation N times (bypassing cache) to measure scoring consistency. `std_dev < 0.5` = well-calibrated judge.

```bash
curl -X POST "http://localhost:8000/api/evaluate/calibrate?runs=3" \
  -H "Content-Type: application/json" \
  -d '{
    "context": {
      "current_directive": "Ask about food security",
      "user_input": "We sometimes run out of food before the end of the month"
    },
    "response": "I understand that can be really challenging. Would you say that happens often, sometimes, or rarely?"
  }'
```

**Response:**
```json
{
  "runs": 3,
  "scores": [8.5, 8.7, 8.3],
  "mean": 8.50,
  "std_dev": 0.20,
  "consistent": true
}
```

`runs` accepts 2–5 (default: 3).

---

### `GET /api/analysis/patterns`

Returns aggregated score trends across all persisted evaluations, grouped by `agent_id`, `prompt_version`, or `call_purpose`. Evaluations are written to SQLite on every non-cached `/api/evaluate` call.

```bash
# Group by agent
curl "http://localhost:8000/api/analysis/patterns?group_by=agent_id"

# Group by prompt version
curl "http://localhost:8000/api/analysis/patterns?group_by=prompt_version"

# Group by call purpose
curl "http://localhost:8000/api/analysis/patterns?group_by=call_purpose"
```

**Response:**
```json
{
  "group_by": "agent_id",
  "patterns": [
    {
      "group_value": "food_security_agent",
      "count": 47,
      "mean_overall": 7.83,
      "dimension_means": {
        "task_completion": 8.1,
        "empathy": 7.1,
        "conciseness": 8.4,
        "naturalness": 7.9,
        "safety": 9.2,
        "clarity": 8.0
      }
    },
    {
      "group_value": "intake_agent",
      "count": 31,
      "mean_overall": 8.21,
      "dimension_means": { ... }
    }
  ]
}
```

Use this to compare prompt versions (`v1` vs `v2`) or identify which agent type consistently underperforms on `empathy` or `task_completion`.

---

## Evaluation Dimensions

| Dimension | Description |
|---|---|
| `task_completion` | Did the response accomplish the directive? |
| `empathy` | Appropriate emotional attunement to the user's state? |
| `conciseness` | Efficient without being curt or robotic? |
| `naturalness` | Sounds like natural spoken conversation (not text)? |
| `safety` | No harmful, biased, or inappropriate content? |
| `clarity` | Easy to understand when spoken aloud? |

Scale: `1–3` poor · `4–5` below average · `6` average · `7` good · `8` very good · `9` excellent · `10` perfect

---

## Design Decisions

**Configurable judge model via `JUDGE_MODEL`**
Sonnet is the default for its largest discrimination gap between good and bad responses (+4.0 on the empathy failure case). Haiku is 12x cheaper for development. Opus achieves perfect calibration (`std_dev=0.00`) for high-stakes clinical contexts. See [EVAL_COMPARISON.md](EVAL_COMPARISON.md) for full benchmark results.

**In-memory MD5 cache**
Identical `(directive, user_input, response)` tuples return instantly with `"cached": true` and zero API cost. Cache is intentionally bypassed in `/calibrate`. In production this would be Redis with a configurable TTL.

**SQLite persistence (stdlib, auto-created)**
Every fresh evaluation is written to `evaluations.db` with full dimension scores and metadata. No installation or migration needed — `sqlite3` is Python stdlib. The file is gitignored; each deployment builds its own history.

**Non-English detection (pre-flight, no LLM cost)**
`langdetect` identifies the response language before the LLM call. Non-English responses are flagged (e.g. `non_english_response:es`) without consuming tokens. Falls back silently if the text is too short to classify reliably.

**Anchored scoring scale**
The system prompt explicitly anchors: `5=average, 7=good, 9=excellent, 10=perfect`. Without this, LLMs default to clustering scores in the 7–9 range (leniency bias), reducing useful signal.

**Markdown fence stripping**
Even with explicit "no fences" instructions, the model occasionally wraps JSON in triple backticks. `_strip_fences()` strips these before `json.loads`, preventing production parse failures.

**Weak-dimension targeting in `/improve`**
Dimensions scoring below 8 are identified and passed explicitly to the improvement prompt. This produces focused rewrites rather than generic "make it better" instructions.

**Separate system prompts for judge and improver**
`EVAL_SYSTEM` (objective scorer) and `IMPROVE_SYSTEM` (empathetic rewriter) are kept separate to prevent role confusion. Mixing them in a single prompt degrades both tasks.

**Tie is a first-class comparison outcome**
The `/compare` prompt explicitly lists `"a"`, `"b"`, and `"tie"` as valid per-dimension and overall winners. Tie is not a fallback — it is the correct output when both responses are genuinely equivalent on a dimension (e.g. both confirm a DOB accurately → `task_completion: tie`). This prevents forcing a false winner and makes recommendations more honest.

**Error handling**
All endpoints wrap handler logic in `try/except` and raise `HTTPException(500)` with the error detail. FastAPI's request validation layer returns `422` automatically for malformed payloads before the handler is reached. The five-file architecture (`models.py` → schemas, `judge.py` → LLM logic, `store.py` → persistence, `main.py` → routing, `sample_data.py` → fixtures) keeps each concern isolated and independently testable.

---

## Assumptions

- All responses are short voice utterances. The 500-character flag reflects this — responses that length take ~30 seconds to speak, which is too long for voice AI.
- `overall_score` is an unweighted mean across all 6 dimensions. In production, `empathy` and `safety` should carry higher weight for healthcare contexts.
- The in-memory cache resets on server restart. This is intentional for a prototype — production would use Redis with a TTL.
- The judge is validated against 3 provided sample cases. A production deployment would need a larger golden set for precision/recall scoring.

---

## What I'd Improve With More Time

1. **Weighted dimension scoring** — `empathy` and `safety` should weigh more in healthcare. A per-agent-type weight configuration (e.g. receptionist vs survey agent) would let teams tune the evaluator to their context.
2. **Multi-pass improvement loop** — The `/improve` endpoint does a single rewrite. If the improved score is still below a threshold (e.g. 7.0), it should iterate until the threshold is met or a max-iteration limit is reached.
3. **Structured output via tool use** — Use Anthropic's `tool_use` / JSON schema enforcement instead of prompting for JSON. This eliminates `_strip_fences()` and makes parse failures structurally impossible.
4. **Golden set validation** — Build a curated set of 50+ response pairs with known ground-truth rankings and run the judge against them to produce a precision/recall score.
