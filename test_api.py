from app import app


def test_ping():
    client = app.test_client()
    response = client.post(
        "/predict",
        json={"features": [5.1, 3.5, 1.4, 0.2]}
    )
    assert response.status_code == 200
    data = response.get_json()
    assert "prediction" in data


def test_predict_batch():
    client = app.test_client()
    features = [
        [5.1, 3.5, 1.4, 0.2],
        [4.9, 3.0, 1.4, 0.2],
        [4.7, 3.2, 1.3, 0.2],
        [7.0, 3.2, 4.7, 1.4],
        [6.4, 3.2, 4.5, 1.5],
        [6.9, 3.1, 4.9, 1.5],
        [6.3, 3.3, 6.0, 2.5],
        [5.8, 2.7, 5.1, 1.9],
        [7.1, 3.0, 5.9, 2.1],
        [5.0, 3.6, 1.4, 0.2],
    ]
    response = client.post("/predict_batch", json={"features": features})
    assert response.status_code == 200
    predictions = response.get_json()["predictions"]
    assert len(predictions) == 10
