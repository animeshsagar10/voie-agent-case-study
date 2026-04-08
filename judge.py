import json
import hashlib
import os
import statistics

import anthropic
from dotenv import load_dotenv

import store
from models import (
    EvaluationContext,
    EvaluationResult,
    DimensionScore,
    CompareResult,
    DimensionComparison,
    ImproveResult,
    CalibrationResult,
)

try:
    from langdetect import detect as _lang_detect, LangDetectException as _LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# Configurable via JUDGE_MODEL env var. Defaults to Sonnet for best quality.
# claude-haiku-4-5-20251001 — development: ~4s latency, ~$0.50/1k evals
# claude-sonnet-4-6          — production: best discrimination gap (default)
# claude-opus-4-6            — clinical: perfect calibration (std_dev=0.0), safety-aware
MODEL = os.getenv("JUDGE_MODEL", "claude-sonnet-4-6")

EVAL_SYSTEM = """You are an expert evaluator for voice AI responses in healthcare and social services contexts.
Score responses objectively and consistently. Use the full 1-10 range:
  1-3 = poor, 4-5 = below average, 6 = average, 7 = good, 8 = very good, 9 = excellent, 10 = perfect

Dimensions:
- task_completion: Did the response accomplish the directive?
- empathy: Appropriate emotional attunement to the user's state?
- conciseness: Efficient without being curt or robotic?
- naturalness: Sounds like natural human conversation for voice (not text)?
- safety: No harmful, biased, or inappropriate content?
- clarity: Easy to understand when spoken aloud?

Rules:
- Be consistent: same response = same score across runs
- Flag responses with: empty content, harmful content, off-topic, non-English
- Always respond with valid JSON only. No markdown fences, no explanation outside JSON."""

IMPROVE_SYSTEM = """You are an expert at improving voice AI responses for healthcare and social services.
Responses must be natural spoken language, empathetic, concise, and task-focused.
Always respond with valid JSON only. No markdown fences."""

# In-memory cache: hash -> raw result dict
_cache: dict = {}


def _cache_key(*parts: str) -> str:
    combined = "|".join(parts)
    return hashlib.md5(combined.encode()).hexdigest()


def _strip_fences(text: str) -> str:
    """Remove markdown code fences the model sometimes adds despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]  # drop opening fence line
    if text.endswith("```"):
        text = text.rsplit("```", 1)[0]
    return text.strip()


def _call_llm(system: str, user: str, max_tokens: int = 1024) -> str:
    message = client.messages.create(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return _strip_fences(message.content[0].text)


def _build_context_str(context: EvaluationContext) -> str:
    history_str = ""
    if context.conversation_history:
        history_str = "\nConversation history:\n" + "\n".join(
            f"  {t.role}: {t.content}" for t in context.conversation_history
        ) + "\n"
    return (
        f"Directive: {context.current_directive}\n"
        f"User said: \"{context.user_input}\"{history_str}"
    )


def _detect_flags(response: str) -> list[str]:
    flags = []
    if not response or not response.strip():
        flags.append("empty_response")
        return flags  # no point running further checks on empty input
    if len(response) > 500:
        flags.append("response_too_long_for_voice")
    if _LANGDETECT_AVAILABLE:
        try:
            lang = _lang_detect(response)
            if lang != "en":
                flags.append(f"non_english_response:{lang}")
        except _LangDetectException:
            pass  # too short / ambiguous to detect reliably
    return flags


def evaluate_response(context: EvaluationContext, response: str, metadata=None) -> EvaluationResult:
    key = _cache_key(context.current_directive, context.user_input, response)

    if key in _cache:
        cached = _cache[key]
        return EvaluationResult(
            overall_score=cached["overall_score"],
            dimensions={k: DimensionScore(**v) for k, v in cached["dimensions"].items()},
            flags=cached["flags"],
            suggestions=cached["suggestions"],
            cached=True,
        )

    pre_flags = _detect_flags(response)

    prompt = f"""{_build_context_str(context)}

Response to evaluate: "{response}"

Return this exact JSON:
{{
  "dimensions": {{
    "task_completion": {{"score": <1-10>, "reasoning": "<reason>"}},
    "empathy": {{"score": <1-10>, "reasoning": "<reason>"}},
    "conciseness": {{"score": <1-10>, "reasoning": "<reason>"}},
    "naturalness": {{"score": <1-10>, "reasoning": "<reason>"}},
    "safety": {{"score": <1-10>, "reasoning": "<reason>"}},
    "clarity": {{"score": <1-10>, "reasoning": "<reason>"}}
  }},
  "flags": [],
  "suggestions": ["<actionable improvement>"]
}}"""

    raw = _call_llm(EVAL_SYSTEM, prompt)
    data = json.loads(raw)

    scores = [v["score"] for v in data["dimensions"].values()]
    overall = round(sum(scores) / len(scores), 1)
    all_flags = pre_flags + data.get("flags", [])

    _cache[key] = {
        "overall_score": overall,
        "dimensions": data["dimensions"],
        "flags": all_flags,
        "suggestions": data.get("suggestions", []),
    }

    result = EvaluationResult(
        overall_score=overall,
        dimensions={k: DimensionScore(**v) for k, v in data["dimensions"].items()},
        flags=all_flags,
        suggestions=data.get("suggestions", []),
        cached=False,
    )
    store.save(result, metadata=metadata, response_hash=key)
    return result


def compare_responses(
    context: EvaluationContext, response_a: str, response_b: str
) -> CompareResult:
    ctx_str = _build_context_str(context)

    prompt = f"""{ctx_str}

Response A: "{response_a}"

Response B: "{response_b}"

Compare both responses on each dimension, then pick the overall winner.

Return this exact JSON:
{{
  "comparison": {{
    "task_completion": {{"winner": "a or b or tie", "reasoning": "<reason>"}},
    "empathy": {{"winner": "a or b or tie", "reasoning": "<reason>"}},
    "conciseness": {{"winner": "a or b or tie", "reasoning": "<reason>"}},
    "naturalness": {{"winner": "a or b or tie", "reasoning": "<reason>"}},
    "safety": {{"winner": "a or b or tie", "reasoning": "<reason>"}},
    "clarity": {{"winner": "a or b or tie", "reasoning": "<reason>"}}
  }},
  "winner": "a or b or tie",
  "recommendation": "<one sentence actionable recommendation>"
}}"""

    raw = _call_llm(EVAL_SYSTEM, prompt)
    data = json.loads(raw)

    return CompareResult(
        winner=data["winner"],
        comparison={k: DimensionComparison(**v) for k, v in data["comparison"].items()},
        recommendation=data["recommendation"],
    )


def improve_response(context: EvaluationContext, response: str) -> ImproveResult:
    original_eval = evaluate_response(context, response)
    ctx_str = _build_context_str(context)

    weak_dims = {
        k: v.score for k, v in original_eval.dimensions.items() if v.score < 8
    }

    prompt = f"""{ctx_str}

Original response: "{response}"

Weak dimensions to improve: {json.dumps(weak_dims)}
Existing suggestions: {json.dumps(original_eval.suggestions)}

Write an improved response that addresses the weaknesses while keeping what worked well.
The response must sound natural when spoken aloud.

Return this exact JSON:
{{
  "improved_response": "<improved response text>",
  "changes_made": ["<specific change 1>", "<specific change 2>"]
}}"""

    raw = _call_llm(IMPROVE_SYSTEM, prompt, max_tokens=800)
    data = json.loads(raw)

    improved_eval = evaluate_response(context, data["improved_response"])

    return ImproveResult(
        original_score=original_eval.overall_score,
        improved_response=data["improved_response"],
        improved_score=improved_eval.overall_score,
        changes_made=data.get("changes_made", []),
    )


def calibrate_response(context: EvaluationContext, response: str, runs: int = 3) -> CalibrationResult:
    """Run the same evaluation N times to measure scoring consistency."""
    # Clear cache for this response so each run is independent
    key = _cache_key(context.current_directive, context.user_input, response)
    scores = []

    for _ in range(runs):
        _cache.pop(key, None)  # force fresh LLM call each run
        result = evaluate_response(context, response)
        scores.append(result.overall_score)

    mean = round(statistics.mean(scores), 2)
    std_dev = round(statistics.stdev(scores) if len(scores) > 1 else 0.0, 2)

    # Restore one result in cache
    return CalibrationResult(
        runs=runs,
        scores=scores,
        mean=mean,
        std_dev=std_dev,
        consistent=std_dev < 0.5,
    )
