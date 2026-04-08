import statistics

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import judge
import store
from models import (
    AggregateStats,
    BatchEvaluateRequest,
    BatchEvaluateResult,
    CalibrationResult,
    CompareRequest,
    CompareResult,
    EvaluateRequest,
    EvaluationResult,
    ImproveRequest,
    ImproveResult,
    PatternEntry,
    PatternResult,
)

app = FastAPI(
    title="LLM Response Quality Evaluator",
    description="Automated multi-dimensional evaluation of voice AI responses using LLM-as-Judge.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/evaluate", response_model=EvaluationResult)
def evaluate(request: EvaluateRequest):
    """Evaluate a single voice AI response across 6 quality dimensions."""
    try:
        return judge.evaluate_response(request.context, request.response, metadata=request.metadata)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/batch", response_model=BatchEvaluateResult)
def evaluate_batch(request: BatchEvaluateRequest):
    """Evaluate multiple responses and return individual scores + aggregate statistics."""
    if not request.evaluations:
        raise HTTPException(status_code=400, detail="evaluations list cannot be empty")
    try:
        results = [
            judge.evaluate_response(r.context, r.response, metadata=r.metadata)
            for r in request.evaluations
        ]

        overall_scores = [r.overall_score for r in results]
        dim_scores: dict[str, list] = {}
        for r in results:
            for dim, score_obj in r.dimensions.items():
                dim_scores.setdefault(dim, []).append(score_obj.score)

        aggregate = AggregateStats(
            count=len(results),
            mean_overall=round(statistics.mean(overall_scores), 2),
            min_overall=min(overall_scores),
            max_overall=max(overall_scores),
            dimension_means={
                dim: round(statistics.mean(scores), 2)
                for dim, scores in dim_scores.items()
            },
        )

        return BatchEvaluateResult(results=results, aggregate=aggregate)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/compare", response_model=CompareResult)
def compare(request: CompareRequest):
    """Compare two responses (A/B test) against the same context."""
    try:
        return judge.compare_responses(
            request.context, request.response_a, request.response_b
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/improve", response_model=ImproveResult)
def improve(request: ImproveRequest):
    """Generate an improved version of a response and score both."""
    try:
        return judge.improve_response(request.context, request.response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/evaluate/calibrate", response_model=CalibrationResult)
def calibrate(request: EvaluateRequest, runs: int = Query(default=3, ge=2, le=5)):
    """
    Bonus: Run the same evaluation N times to measure scoring consistency.
    std_dev < 0.5 indicates the judge is well-calibrated.
    """
    try:
        return judge.calibrate_response(request.context, request.response, runs=runs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/analysis/patterns", response_model=PatternResult)
def analysis_patterns(
    group_by: str = Query(
        default="agent_id",
        description="Group evaluations by: agent_id | prompt_version | call_purpose",
        pattern="^(agent_id|prompt_version|call_purpose)$",
    )
):
    """
    Bonus: Aggregated score trends grouped by agent_id, prompt_version, or call_purpose.
    Evaluations are persisted to evaluations.db on every /api/evaluate call.
    """
    try:
        patterns = store.get_patterns(group_by)
        return PatternResult(
            group_by=group_by,
            patterns=[PatternEntry(**p) for p in patterns],
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
