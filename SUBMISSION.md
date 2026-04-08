# Case Study Submission — LLM Response Quality Evaluator
**Candidate:** Animesh Sagar
**GitHub:** https://github.com/animeshsagar10/voie-agent-case-study
**Case Study:** #3 — LLM Response Quality Evaluator

---

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Add your ANTHROPIC_API_KEY and optionally set JUDGE_MODEL

# Start the server
uv run python main.py
# → API live at http://localhost:8000
# → Interactive docs at http://localhost:8000/docs

# Run tests
uv sync --extra dev
uv run pytest tests/ -v
```

---

## What Was Built

A FastAPI service that evaluates voice AI responses using the **LLM-as-Judge** pattern. Claude acts as the evaluator, scoring responses across 6 dimensions relevant to healthcare voice AI: task completion, empathy, conciseness, naturalness, safety, and clarity.

### Endpoints

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/evaluate` | POST | Score a single response across 6 dimensions |
| `/api/evaluate/batch` | POST | Score multiple responses + aggregate statistics |
| `/api/compare` | POST | A/B comparison of two responses to the same context |
| `/api/improve` | POST | Generate an improved version and score both |
| `/api/evaluate/calibrate` | POST | Measure scoring consistency across N independent runs |
| `/health` | GET | Health check |

---

## Design Decisions and Tradeoffs

### 1. LLM-as-Judge with configurable model

The judge model is selected via the `JUDGE_MODEL` environment variable, defaulting to `claude-sonnet-4-6`. This was a deliberate decision driven by benchmarking (see full results below).

The key finding: **Sonnet discriminates significantly harder on empathy failures.** On the food security case (`eval_002`), it scored the cold, dismissive response **4.3** vs Haiku's **6.2**. A judge that scores a clearly empathy-failing healthcare response 6.2/10 is not useful — it fails to flag the responses that most need human review or improvement. Sonnet's stricter scoring is the correct behaviour for this use case.

```bash
JUDGE_MODEL=claude-haiku-4-5-20251001  # development — 12x cheaper, 2x faster
JUDGE_MODEL=claude-sonnet-4-6          # production (default)
JUDGE_MODEL=claude-opus-4-6            # high-stakes / clinical — perfect calibration
```

### 2. Anchored scoring scale

The system prompt explicitly anchors the scale: `5=average, 7=good, 9=excellent, 10=perfect`. Without anchoring, LLMs default to clustering scores between 7–9, reducing the signal. This is a known LLM-as-Judge bias ("leniency bias") and anchoring is the standard mitigation.

### 3. In-memory cache keyed on MD5(directive + user_input + response)

Identical requests return immediately with `"cached": true` and zero additional API cost. The cache is intentionally bypassed in the `/calibrate` endpoint which requires independent runs. In production this would be Redis with a TTL.

### 4. Markdown fence stripping

Even with explicit `"no markdown fences"` instructions in the system prompt, the model occasionally wraps JSON in triple backticks. `_strip_fences()` handles this defensively before `json.loads`, preventing parse failures in production.

### 5. Pre-flight flag detection (no LLM call needed)

Empty responses and responses over 500 characters are flagged client-side before the LLM call. This avoids wasting tokens on trivially bad inputs and ensures the API always returns a useful response even if the input is degenerate.

### 6. Weak-dimension targeting in `/improve`

The improvement endpoint identifies dimensions scoring below 8 and passes them explicitly to the improvement prompt. This produces focused rewrites rather than generic "make it better" outputs. The result: Sonnet improved the worst bad response by **+4.4 points** (4.3 → 8.7).

### 7. Separate system prompts for judge vs improver roles

`EVAL_SYSTEM` and `IMPROVE_SYSTEM` are kept separate to prevent role confusion. The judge scores objectively; the improver writes empathetically. Mixing them in a single prompt degrades both tasks.

---

## Benchmark Results — Haiku vs Sonnet vs Opus

All experiments run on the 3 sample cases from the case study spec. Full detail in [`EVAL_COMPARISON.md`](EVAL_COMPARISON.md).

### Evaluation Scores (Good vs Bad Responses)

| Case | Context | Model | Good ↑ | Bad ↓ | Gap | Discriminates? |
|---|---|---|---|---|---|---|
| eval_001 | DOB verification | Haiku | 8.3 | 8.5 | -0.2 | ❌ |
| eval_001 | DOB verification | Sonnet | 8.3 | 7.7 | +0.6 | ✅ |
| eval_001 | DOB verification | Opus | 8.3 | 7.8 | +0.5 | ✅ |
| eval_002 | Food security / empathy | Haiku | 8.8 | 6.2 | +2.6 | ✅ |
| eval_002 | Food security / empathy | Sonnet | 8.3 | 4.3 | **+4.0** | ✅ |
| eval_002 | Food security / empathy | Opus | 8.3 | 5.0 | +3.3 | ✅ |
| eval_003 | Survey confusion | Haiku | 8.8 | 6.8 | +2.0 | ✅ |
| eval_003 | Survey confusion | Sonnet | 9.0 | 4.3 | **+4.7** | ✅ |
| eval_003 | Survey confusion | Opus | 9.0 | 5.3 | +3.7 | ✅ |

> **Key insight:** Haiku was the only model to fail discrimination on eval_001. Sonnet has the largest gaps overall — strictest judge. Opus's standout finding: it scored the bad food security response `safety: 4`, flagging that dismissing someone mid-food-crisis is a clinical safety concern — the most sophisticated signal of all three models.

### Dimension Averages — Good Responses (across all 3 cases)

| Dimension | Haiku | Sonnet | Opus |
|---|---|---|---|
| Task Completion | 9.0 | 8.0 | 8.3 |
| Empathy | 8.0 | 7.7 | 7.3 |
| Conciseness | 8.0 | 9.0 | 9.0 |
| Naturalness | 8.0 | 8.7 | 8.7 |
| Safety | **10.0** | 9.3 | 9.3 |
| Clarity | 9.0 | 8.7 | 9.0 |

> Haiku scores Safety 10/10 across the board (ceiling effect — no room for nuance). Sonnet and Opus leave calibrated headroom.

### A/B Comparison Accuracy

| Case | Expected | Haiku | ✓ | Sonnet | ✓ | Opus | ✓ |
|---|---|---|---|---|---|---|---|
| eval_001 | A (concise) | A | ✅ | A | ✅ | A | ✅ |
| eval_002 | A (empathetic) | A | ✅ | A | ✅ | A | ✅ |
| eval_003 | A (reassuring) | A | ✅ | A | ✅ | A | ✅ |

**All three models: 3/3 (100%) accuracy.** Opus recommendations cited clinical frameworks by name (Hunger Vital Sign, USDA Household Food Security Survey) — a level of domain specificity directly useful for healthcare prompt engineering.

### Improvement Quality (eval_002 — food security dismissal)

| Model | Original Score | Improved Score | Delta | Latency |
|---|---|---|---|---|
| Haiku | 6.2 | 8.8 | +2.6 | 7.1s |
| Sonnet | 4.3 | 8.7 | **+4.4** | 16.8s |
| Opus | 5.0 | 8.5 | +3.5 | 18.0s |

**Opus improved response:**
> *"I'm really sorry you've been going through that. Thank you for telling me — that takes courage. What you've shared tells me food has been a serious concern, and I want to make sure we get you connected to some help right away. Can I ask — within the past twelve months, have you also worried about food running out before you had money to buy more? I just want to make sure we capture the full picture so we can find the right resources for you."*

Sonnet's delta is highest (+4.4) because its starting score was lowest (4.3) — it correctly identified the original as very bad. Opus's response is the most human-sounding of the three.

### Scoring Consistency (Calibration — 3 Independent Runs)

| Model | Run 1 | Run 2 | Run 3 | Mean | Std Dev | Consistent? |
|---|---|---|---|---|---|---|
| Haiku | 8.7 | 8.2 | 8.5 | 8.47 | 0.25 | ✅ |
| Sonnet | 8.5 | 8.7 | 8.3 | 8.50 | 0.20 | ✅ |
| Opus | 8.3 | 8.3 | 8.3 | 8.30 | **0.00** | ✅ |

> **Opus achieved perfect calibration** — identical score across all 3 independent runs. This is the strongest possible signal for A/B test reliability in production.

### Overall Scorecard

| Metric | Haiku | Sonnet | Opus |
|---|---|---|---|
| Avg score — good responses | 8.63 | 8.53 | 8.53 |
| Avg discrimination gap | +1.47 | **+3.10** | +2.50 |
| A/B accuracy | 100% | 100% | 100% |
| Improvement delta | +2.6 | **+4.4** | +3.5 |
| Calibration std dev | 0.25 | 0.20 | **0.00** |
| Avg eval latency | **4.4s** | 9.0s | 11.5s |
| Cost / 1k evals | **~$0.50** | ~$6.00 | ~$42.00 |

### Model Selection Guide

| Use case | Model | Reason |
|---|---|---|
| Development / iteration | **Haiku** | 4.4s latency, $0.50/1k evals |
| Production evaluation | **Sonnet** | Best discrimination gap, strong improvement quality |
| High-stakes / clinical | **Opus** | Perfect calibration (std_dev=0.0), safety-aware scoring |

---

## Evaluation Criteria Self-Assessment

### Judge Prompt Design — 30%

- Anchored 1–10 scale reduces leniency bias (`5=average, 7=good, 9=excellent, 10=perfect`)
- Separate system prompts for judge and improver roles prevent role confusion
- `_strip_fences()` makes JSON parsing robust against model formatting drift
- `/api/evaluate/calibrate` endpoint quantifies scoring consistency with std dev — both models score < 0.25, confirming the judge is stable

### Multi-dimensional Analysis — 25%

- All 6 required dimensions implemented: task_completion, empathy, conciseness, naturalness, safety, clarity
- Each dimension returns both a `score` and `reasoning` string — not just a number
- `/api/evaluate/batch` returns `dimension_means` across all responses, enabling pattern analysis by agent or prompt version
- `overall_score` is an unweighted mean across all 6 dimensions (tradeoff noted below)

### Comparison Logic — 20%

- Single LLM call evaluates both responses in the same context window — more consistent than two separate calls
- Returns per-dimension winner (`a` / `b` / `tie`) + overall winner + one-sentence `recommendation`
- Tie is a valid output — explicitly listed in the prompt, not a fallback
- 100% accuracy on all 3 sample cases for both models

### Improvement Generation — 15%

- Weak dimensions (score < 8) are identified and passed explicitly to the improvement prompt
- Both original and improved scores are returned, so the delta is measurable
- `changes_made` lists specific, named actions taken — not generic rewrites
- Sonnet achieved +4.4 improvement delta on the hardest case (food security empathy failure)

### Code Quality — 10%

- 5-file architecture: `models.py` (schemas), `judge.py` (LLM logic), `main.py` (routing), `sample_data.py`, `benchmark.py`
- All endpoints wrapped in `try/except` with proper `HTTPException` responses
- `uv` + `pyproject.toml` for fully reproducible dependency management
- 14 tests covering: happy path, caching, batch, edge cases (empty/long/non-English), comparison, improvement, SQLite persistence, pattern analysis

### Additional Points

| Bonus | Implementation |
|---|---|
| Calibration | `/api/evaluate/calibrate` — runs N independent evals, returns mean, std dev, and `consistent` flag |
| Edge cases | Empty responses flagged pre-LLM (`empty_response`); >500 char responses flagged (`response_too_long_for_voice`); non-English detected via `langdetect` and flagged (`non_english_response:<lang>`) |
| Cost optimisation | MD5-keyed in-memory cache (zero cost on repeat); Haiku for dev / Sonnet for prod via env var; batch endpoint reduces per-request overhead |
| Scoring pattern analysis | SQLite persistence (`evaluations.db`, auto-created) stores every evaluation with `agent_id`, `prompt_version`, `call_purpose`; `/api/analysis/patterns?group_by=agent_id` returns aggregated dimension scores per group |

---

## Assumptions

1. All responses are short voice utterances. The >500 character flag reflects this — responses that length would take ~30 seconds to speak, which is too long for voice AI.
2. `overall_score` is an unweighted mean across 6 dimensions. In production, `empathy` and `safety` should carry higher weight for healthcare contexts.
3. The judge is evaluated on the 3 provided sample cases. A production deployment would need a larger golden set for validation.
4. The cache is in-memory and resets on server restart. This is intentional for a prototype — production would use Redis with a configurable TTL.

---

## What I Would Improve With More Time

1. **Weighted dimension scoring** — `empathy` and `safety` should weigh more in healthcare. A per-agent-type weight configuration (e.g. receptionist vs survey agent) would let teams tune the evaluator to their context.

2. **Multi-pass improvement loop** — The `/improve` endpoint does a single rewrite pass. If the improved score is still below a threshold (e.g. 7.0), it should iterate with the new score and feedback until the threshold is met or a max-iteration limit is reached.

3. **Structured output via tool use** — Use Anthropic's `tool_use` / JSON schema enforcement instead of prompting for JSON. This eliminates the need for `_strip_fences()` and makes parse failures structurally impossible.

4. **Golden set validation** — Build a curated set of 50+ response pairs with known ground-truth quality rankings, and run the judge against them to produce a precision/recall score. This makes the evaluator itself auditable.

5. **Redis cache with TTL** — Replace the in-memory dict with Redis so the cache survives restarts and can be shared across multiple server instances in production.
