def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"


def test_ready(client):
    assert client.get("/ready").status_code == 200


def test_predict_requires_file(client):
    response = client.post("/v1/predict")
    assert response.status_code == 422


def test_predict_rejects_invalid_image_file(client, invalid_image_file):
    response = client.post(
        "/v1/predict",
        files=invalid_image_file,
    )
    assert response.status_code == 422


def test_predict_rejects_unsupported_content_type(client, unsupported_file):
    response = client.post(
        "/v1/predict",
        files=unsupported_file,
    )
    assert response.status_code == 415