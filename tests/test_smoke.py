from asl.config import settings


def test_predict_smoke(client, valid_image_file):
    response = client.post(
        "/v1/predict",
        files = valid_image_file,
    )
    assert response.status_code == 200
    body = response.json()
    assert body['prediction_class'] in settings.class_names
    assert 0.0 <= body['confidence'] <= 1.0
    assert len(body['all_probabilities']) == len(settings.class_names)
    print(body['all_probabilities'])
    assert sum(body['all_probabilities'].values()) <= 1.0 + 1e-12
    assert body["latency_ms"] >= 0
    assert body["model_version"]


def test_batch_and_single_agree(client, valid_image_file):
    r1 = client.post("/v1/predict", files = valid_image_file)
    r2 = client.post("/v1/predict", files = valid_image_file)
    body1 = r1.json()
    body2 = r2.json()
    assert body1['prediction_class'] == body2['prediction_class']
    assert abs(body1['confidence'] - body2['confidence']) < 1e-12