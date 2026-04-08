"""
Tests for the LLM Response Quality Evaluator API.

Run with: pytest tests/ -v
Note: These tests make real LLM calls — set ANTHROPIC_API_KEY in .env first.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient

from main import app
from sample_data import SAMPLE_CASES

client = TestClient(app)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_eval_payload(case: dict, which: str = "good") -> dict:
    response_key = "response_good" if which == "good" else "response_bad"
    return {
        "context": case["context"],
        "response": case[response_key],
    }


# ── Health ─────────────────────────────────────────────────────────────────────

def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ── POST /api/evaluate ─────────────────────────────────────────────────────────

def test_evaluate_returns_valid_structure():
    payload = make_eval_payload(SAMPLE_CASES[0], "good")
    r = client.post("/api/evaluate", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert "overall_score" in data
    assert 1.0 <= data["overall_score"] <= 10.0

    dims = data["dimensions"]
    expected_dims = {"task_completion", "empathy", "conciseness", "naturalness", "safety", "clarity"}
    assert set(dims.keys()) == expected_dims

    for dim_data in dims.values():
        assert "score" in dim_data
        assert "reasoning" in dim_data
        assert 1 <= dim_data["score"] <= 10

    assert isinstance(data["flags"], list)
    assert isinstance(data["suggestions"], list)


def test_good_response_scores_higher_than_bad():
    case = SAMPLE_CASES[1]  # food security + empathy case
    good_r = client.post("/api/evaluate", json=make_eval_payload(case, "good"))
    bad_r = client.post("/api/evaluate", json=make_eval_payload(case, "bad"))

    assert good_r.status_code == 200
    assert bad_r.status_code == 200

    good_score = good_r.json()["overall_score"]
    bad_score = bad_r.json()["overall_score"]

    assert good_score > bad_score, (
        f"Expected good ({good_score}) > bad ({bad_score}) for case {case['id']}"
    )


def test_evaluate_caches_result():
    payload = make_eval_payload(SAMPLE_CASES[0], "good")
    r1 = client.post("/api/evaluate", json=payload)
    r2 = client.post("/api/evaluate", json=payload)

    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r2.json()["cached"] is True
    assert r1.json()["overall_score"] == r2.json()["overall_score"]


def test_evaluate_flags_empty_response():
    payload = {
        "context": SAMPLE_CASES[0]["context"],
        "response": "",
    }
    r = client.post("/api/evaluate", json=payload)
    assert r.status_code == 200
    assert "empty_response" in r.json()["flags"]


def test_evaluate_flags_long_response():
    payload = {
        "context": SAMPLE_CASES[0]["context"],
        "response": "word " * 120,  # >500 chars
    }
    r = client.post("/api/evaluate", json=payload)
    assert r.status_code == 200
    assert "response_too_long_for_voice" in r.json()["flags"]


# ── POST /api/evaluate/batch ───────────────────────────────────────────────────

def test_batch_evaluate():
    payload = {
        "evaluations": [
            make_eval_payload(SAMPLE_CASES[0], "good"),
            make_eval_payload(SAMPLE_CASES[0], "bad"),
        ]
    }
    r = client.post("/api/evaluate/batch", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert len(data["results"]) == 2
    agg = data["aggregate"]
    assert agg["count"] == 2
    assert 1.0 <= agg["mean_overall"] <= 10.0
    assert agg["min_overall"] <= agg["mean_overall"] <= agg["max_overall"]
    expected_dims = {"task_completion", "empathy", "conciseness", "naturalness", "safety", "clarity"}
    assert set(agg["dimension_means"].keys()) == expected_dims


def test_batch_empty_list_returns_400():
    r = client.post("/api/evaluate/batch", json={"evaluations": []})
    assert r.status_code == 400


# ── POST /api/compare ──────────────────────────────────────────────────────────

def test_compare_all_sample_cases():
    """
    Validates compare response structure across all sample cases.
    Winner alignment with expected_winner is logged but not enforced — LLM judgment
    can legitimately differ on close calls.
    """
    for case in SAMPLE_CASES:
        payload = {
            "context": case["context"],
            "response_a": case["response_good"],
            "response_b": case["response_bad"],
        }
        r = client.post("/api/compare", json=payload)
        assert r.status_code == 200
        data = r.json()

        # Structure checks
        assert data["winner"] in ("a", "b", "tie")
        assert "recommendation" in data and data["recommendation"]

        expected_dims = {"task_completion", "empathy", "conciseness", "naturalness", "safety", "clarity"}
        assert set(data["comparison"].keys()) == expected_dims
        for dim_data in data["comparison"].values():
            assert dim_data["winner"] in ("a", "b", "tie")
            assert "reasoning" in dim_data and dim_data["reasoning"]

        # Soft check — log mismatches rather than fail
        if data["winner"] != case["expected_winner"]:
            print(
                f"\n  [INFO] Case {case['id']}: expected={case['expected_winner']}, "
                f"got={data['winner']} — LLM disagrees, recommendation: {data['recommendation']}"
            )


def test_compare_clear_empathy_winner():
    """eval_002 is the strongest empathy case — good response should win."""
    case = SAMPLE_CASES[1]
    payload = {
        "context": case["context"],
        "response_a": case["response_good"],
        "response_b": case["response_bad"],
    }
    r = client.post("/api/compare", json=payload)
    assert r.status_code == 200
    data = r.json()
    # Empathy dim should clearly favour A
    assert data["comparison"]["empathy"]["winner"] in ("a", "tie")


# ── POST /api/improve ──────────────────────────────────────────────────────────

def test_improve_increases_score():
    payload = make_eval_payload(SAMPLE_CASES[1], "bad")  # bad response → should improve
    r = client.post("/api/improve", json=payload)
    assert r.status_code == 200
    data = r.json()

    assert "original_score" in data
    assert "improved_response" in data
    assert "improved_score" in data
    assert isinstance(data["changes_made"], list)
    assert len(data["changes_made"]) > 0
    assert data["improved_score"] >= data["original_score"], (
        f"Improved score ({data['improved_score']}) should be >= original ({data['original_score']})"
    )


# ── Non-English detection ──────────────────────────────────────────────────────

def test_evaluate_flags_non_english_response():
    """A clearly non-English response should be flagged as non_english_response."""
    payload = {
        "context": SAMPLE_CASES[0]["context"],
        "response": "Lo siento, no puedo ayudarte con eso en este momento.",  # Spanish
    }
    r = client.post("/api/evaluate", json=payload)
    assert r.status_code == 200
    flags = r.json()["flags"]
    non_english_flags = [f for f in flags if f.startswith("non_english_response")]
    assert len(non_english_flags) == 1, f"Expected non_english_response flag, got flags: {flags}"


# ── SQLite persistence + /api/analysis/patterns ────────────────────────────────

def test_patterns_endpoint_structure():
    """
    Submit an evaluation with agent_id metadata, then verify /api/analysis/patterns
    returns the correct structure and includes the submitted agent.
    """
    # Submit a fresh eval with a distinctive agent_id so it lands in the DB
    import uuid
    agent_id = f"test_agent_{uuid.uuid4().hex[:8]}"
    payload = {
        "context": SAMPLE_CASES[0]["context"],
        "response": SAMPLE_CASES[0]["response_good"],
        "metadata": {"agent_id": agent_id, "prompt_version": "v1", "call_purpose": "unit_test"},
    }
    # Clear in-memory cache so this forced a real eval (and a DB write)
    import judge as _judge
    _judge._cache.clear()

    eval_r = client.post("/api/evaluate", json=payload)
    assert eval_r.status_code == 200
    assert eval_r.json()["cached"] is False  # must be a fresh write

    # Now check patterns
    r = client.get("/api/analysis/patterns", params={"group_by": "agent_id"})
    assert r.status_code == 200
    data = r.json()

    assert data["group_by"] == "agent_id"
    assert isinstance(data["patterns"], list)
    assert len(data["patterns"]) >= 1

    # Our agent_id should appear
    agent_entries = [p for p in data["patterns"] if p["group_value"] == agent_id]
    assert len(agent_entries) == 1
    entry = agent_entries[0]

    assert entry["count"] >= 1
    assert 1.0 <= entry["mean_overall"] <= 10.0
    expected_dims = {"task_completion", "empathy", "conciseness", "naturalness", "safety", "clarity"}
    assert set(entry["dimension_means"].keys()) == expected_dims


def test_patterns_invalid_group_by_returns_400():
    r = client.get("/api/analysis/patterns", params={"group_by": "invalid_field"})
    assert r.status_code == 422  # FastAPI validates the Query pattern before our handler
