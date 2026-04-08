from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class ConversationTurn(BaseModel):
    role: str
    content: str


class EvaluationContext(BaseModel):
    conversation_history: Optional[List[ConversationTurn]] = []
    current_directive: str
    user_input: str


class EvaluationMetadata(BaseModel):
    agent_id: Optional[str] = None
    prompt_version: Optional[str] = None
    model: Optional[str] = None


class EvaluateRequest(BaseModel):
    context: EvaluationContext
    response: str
    metadata: Optional[EvaluationMetadata] = None


class DimensionScore(BaseModel):
    score: float
    reasoning: str


class EvaluationResult(BaseModel):
    overall_score: float
    dimensions: Dict[str, DimensionScore]
    flags: List[str]
    suggestions: List[str]
    cached: bool = False


class BatchEvaluateRequest(BaseModel):
    evaluations: List[EvaluateRequest]


class AggregateStats(BaseModel):
    count: int
    mean_overall: float
    min_overall: float
    max_overall: float
    dimension_means: Dict[str, float]


class BatchEvaluateResult(BaseModel):
    results: List[EvaluationResult]
    aggregate: AggregateStats


class CompareRequest(BaseModel):
    context: EvaluationContext
    response_a: str
    response_b: str


class DimensionComparison(BaseModel):
    winner: str  # "a", "b", or "tie"
    reasoning: str


class CompareResult(BaseModel):
    winner: str  # "a", "b", or "tie"
    comparison: Dict[str, DimensionComparison]
    recommendation: str


class ImproveRequest(BaseModel):
    context: EvaluationContext
    response: str
    metadata: Optional[EvaluationMetadata] = None


class ImproveResult(BaseModel):
    original_score: float
    improved_response: str
    improved_score: float
    changes_made: List[str]


class CalibrationResult(BaseModel):
    runs: int
    scores: List[float]
    mean: float
    std_dev: float
    consistent: bool  # True if std_dev < 0.5
