import psycopg
from psycopg.types.json import Json

from asl.config import settings

DDL = """
CREATE TABLE IF NOT EXISTS predictions

    request_id          uuid PRIMARY KEY,
    ts                  timestamptz NOT NULL DEFAULT now(),
    model_version       text NOT NULL,
    prediction_class    text NOT NULL,
    all_probabilities   jsonb NOT NULL,
    input_metadata      jsonb NOT NULL,
    latency_ms          real
"""


def init() -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("SELECT pg_advisory_xact_lock(7001)")
        conn.execute(DDL)


def save_prediction(request_id: str, 
                    model_version: str,
                    prediction_class: str, 
                    all_probabilities: dict[float],
                    input_metadata: dict,
                    latency_ms: float, 
                    ) -> None:
    if not settings.database_url:
        return
    with psycopg.connect(settings.database_url) as conn:
        conn.execute(
            "INSERT INTO predictions (request_id, model_version, prediction_class, all_probabilities, input_metadata, latency_ms) "
            "VALUES (%s, %s, %s, %s, %s, %s)",
            (request_id, model_version, prediction_class, Json(all_probabilities), Json(input_metadata), latency_ms),
        )