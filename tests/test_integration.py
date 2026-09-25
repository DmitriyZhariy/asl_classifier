import os

import psycopg
import pytest

DATABASE_URL = os.getenv("DATABASE_URL")

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not DATABASE_URL, reason="no Postgres database URL provided")
]


def test_prediction_is_logged(client, valid_image_file):
    response = client.post(
            "/v1/predict",
            files = valid_image_file,
        )

    body = response.json()

    with psycopg.connect(DATABASE_URL) as conn:
        row = conn.execute(
            "SELECT model_version, prediction_class, latency_ms, all_probabilities " \
            "FROM predictions WHERE request_id = %s",
            (body["request_id"],),
        ).fetchone()

    assert row is not None
    assert row[0] == body["model_version"]
    assert row[1] == body["prediction_class"]
    assert row[2] == pytest.approx(body["latency_ms"])
    assert row[3] == body["all_probabilities"]