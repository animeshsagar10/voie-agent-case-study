# LLM Response Quality Evaluator

Automated multi-dimensional evaluation of voice AI responses using the **LLM-as-Judge** pattern. Built for BloomingHealth's voice AI platform to replace manual call quality review.

## Setup

**Requirements:** Python 3.11+, [uv](https://docs.astral.sh/uv/)

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Add your Anthropic API key
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env

# Start the server
uv run python main.py
```

API is now live at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

## Run Tests

```bash
uv sync --extra dev
uv run pytest tests/ -v
```

All 11 tests pass, covering evaluation, caching, batch processing, A/B comparison, improvement, and edge cases.

---

## API Endpoints

### `POST /api/evaluate`
Scores a single response across 6 dimensions (1–10 scale).

```bash
curl -X POST http://localhost:8000/api/evaluate \
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
  "overall_score": 8.5,
  "dimensions": {
    "task_completion": { "score": 9, "reasoning": "Successfully asked clarifying question" },
    "empathy":         { "score": 8, "reasoning": "Acknowledged difficulty appropriately" },
    "conciseness":     { "score": 9, "reasoning": "Brief and focused" },
    "naturalness":     { "score": 8, "reasoning": "Conversational tone" },
    "safety":          { "score": 10, "reasoning": "No harmful content" },
    "clarity":         { "score": 8, "reasoning": "Clear and easy to follow" }
  },
  "flags": [],
  "suggestions": ["Consider warmer acknowledgment before the clarifying question"],
  "cached": false
}
```

---

### `POST /api/evaluate/batch`
Evaluates multiple responses and returns aggregate statistics.

```bash
curl -X POST http://localhost:8000/api/evaluate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "evaluations": [
      { "context": { "current_directive": "...", "user_input": "..." }, "response": "Response A" },
      { "context": { "current_directive": "...", "user_input": "..." }, "response": "Response B" }
    ]
  }'
```

Returns individual scores + `aggregate` with `mean_overall`, `min/max`, and per-dimension means.

---

### `POST /api/compare`
A/B comparison of two responses to the same context.

```bash
curl -X POST http://localhost:8000/api/compare \
  -H "Content-Type: application/json" \
  -d '{
    "context": { "current_directive": "Verify date of birth", "user_input": "March 15, 1985" },
    "response_a": "Got it, March 15th, 1985. Thank you.",
    "response_b": "Perfect! I have recorded your date of birth. Is there anything else?"
  }'
```

**Response:**
```json
{
  "winner": "a",
  "comparison": {
    "conciseness": { "winner": "a", "reasoning": "Response A is appropriately brief" },
    "task_completion": { "winner": "tie", "reasoning": "Both confirm the DOB" },
    ...
  },
  "recommendation": "Response A is preferred for concise verification scenarios"
}
```

---

### `POST /api/improve`
Generates an improved version of a response and scores both.

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
  "original_score": 5.5,
  "improved_response": "That's a fair question. This survey helps us see if there are resources in your community that might help you — things like food assistance or transportation. Everything you share stays confidential.",
  "improved_score": 8.2,
  "changes_made": ["Added empathetic framing", "Explained purpose in plain language", "Added confidentiality assurance"]
}
```

---

### `POST /api/evaluate/calibrate` *(Bonus)*
Runs the same evaluation N times to measure scoring consistency. `std_dev < 0.5` = well-calibrated.

```bash
curl -X POST "http://localhost:8000/api/evaluate/calibrate?runs=3" \
  -H "Content-Type: application/json" \
  -d '{ "context": { ... }, "response": "..." }'
```

---

## Evaluation Dimensions

| Dimension | Description |
|---|---|
| `task_completion` | Did the response accomplish the directive? |
| `empathy` | Appropriate emotional attunement to user's state? |
| `conciseness` | Efficient without being curt or robotic? |
| `naturalness` | Sounds like natural spoken conversation? |
| `safety` | No harmful, biased, or inappropriate content? |
| `clarity` | Easy to understand when spoken aloud? |

---

## Design Decisions

**LLM-as-Judge with `claude-haiku-4-5`**
Haiku gives fast, cheap evaluations (~$0.001/call) with quality sufficient for structured scoring. The system prompt uses an anchored scale (5=average, 7=good, 9=excellent) to reduce score inflation. The model is swappable — change `MODEL` in `judge.py` for higher fidelity.

**In-memory cache keyed by MD5(directive + user_input + response)**
Identical requests return instantly without re-billing. The cache is intentionally bypassed in the `/calibrate` endpoint to measure true consistency. In production this would be Redis with a TTL.

**Markdown fence stripping**
Even with explicit "no fences" instructions, the model occasionally wraps JSON in triple backticks. `_strip_fences()` handles this defensively before `json.loads`.

**Pre-flight flag detection**
Empty and overly-long responses are flagged client-side before the LLM call, avoiding wasted tokens on trivially bad inputs.

**Improvement uses weak-dimension targeting**
The `/improve` endpoint identifies dimensions scoring below 8 and passes them explicitly to the improvement prompt. This focuses the rewrite rather than asking for generic improvement.

---

## What I'd Improve With More Time

1. **Persistent cache** — Replace the in-memory dict with SQLite or Redis so scores survive restarts and can be queried for analytics.
2. **Scoring patterns dashboard** — Aggregate scores by `agent_id`, `prompt_version`, and `call_purpose` to surface which prompt variants consistently underperform.
3. **Structured output via Anthropic's tool-use** — Use `tool_use` / JSON schema enforcement instead of prompt-based JSON to eliminate parse errors entirely.
4. **Cost optimization** — Batch multiple evaluations into a single LLM call using multi-turn context when evaluating a dataset.
5. **Non-English handling** — Detect language and route to a multilingual model or flag for human review.

---

## Assumptions

- All responses are short voice AI utterances (< ~150 words). Longer responses are flagged but still evaluated.
- The 6 evaluation dimensions are fixed; production would allow dimension sets to be configured per agent type.
- `overall_score` is a simple mean across all 6 dimensions (equal weight). A weighted average could be added per use case.
- The `/improve` endpoint always makes 2–3 LLM calls. In a high-volume setting, improvement generation would be async.
