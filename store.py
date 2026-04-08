"""
SQLite persistence layer for evaluation results.
evaluations.db is created automatically on first run — no setup needed.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "evaluations.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,
    agent_id        TEXT,
    prompt_version  TEXT,
    call_purpose    TEXT,
    overall_score   REAL    NOT NULL,
    task_completion REAL,
    empathy         REAL,
    conciseness     REAL,
    naturalness     REAL,
    safety          REAL,
    clarity         REAL,
    flags           TEXT,
    response_hash   TEXT
)
"""

_DIMS = ["task_completion", "empathy", "conciseness", "naturalness", "safety", "clarity"]


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init():
    with _get_conn() as conn:
        conn.execute(_SCHEMA)


_init()


def save(result, metadata=None, response_hash: str = "") -> None:
    dims = result.dimensions

    def dim_score(name):
        return dims[name].score if name in dims else None

    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO evaluations
               (timestamp, agent_id, prompt_version, call_purpose, overall_score,
                task_completion, empathy, conciseness, naturalness, safety, clarity,
                flags, response_hash)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                datetime.now(timezone.utc).isoformat(),
                metadata.agent_id if metadata else None,
                metadata.prompt_version if metadata else None,
                metadata.call_purpose if metadata else None,
                result.overall_score,
                dim_score("task_completion"),
                dim_score("empathy"),
                dim_score("conciseness"),
                dim_score("naturalness"),
                dim_score("safety"),
                dim_score("clarity"),
                json.dumps(result.flags),
                response_hash,
            ),
        )


def get_patterns(group_by: str) -> list[dict]:
    """Return per-group aggregated scores. group_by: agent_id | prompt_version | call_purpose."""
    allowed = {"agent_id", "prompt_version", "call_purpose"}
    if group_by not in allowed:
        raise ValueError(f"group_by must be one of {sorted(allowed)}")

    dim_avgs = ", ".join(f"AVG({d}) AS {d}" for d in _DIMS)
    sql = f"""
        SELECT
            COALESCE({group_by}, 'unknown') AS group_value,
            COUNT(*)                        AS count,
            AVG(overall_score)              AS mean_overall,
            {dim_avgs}
        FROM evaluations
        GROUP BY {group_by}
        ORDER BY count DESC
    """
    with _get_conn() as conn:
        rows = conn.execute(sql).fetchall()

    result = []
    for row in rows:
        dimension_means = {
            dim: round(row[dim], 2)
            for dim in _DIMS
            if row[dim] is not None
        }
        result.append(
            {
                "group_value": row["group_value"],
                "count": row["count"],
                "mean_overall": round(row["mean_overall"], 2),
                "dimension_means": dimension_means,
            }
        )
    return result
