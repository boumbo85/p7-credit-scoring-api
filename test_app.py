from app import app
import pytest

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert response.data == b"Bienvenu(e) !"

def test_prediction_route(client):
    # Testez la route de prédiction pour un client existant
    response = client.get('/prediction/100031')
    assert response.status_code == 200
    data = response.json
    assert 'ID' in data
    assert 'Prediction' in data
    assert 'Predict_proba' in data
    assert 'Feature_Importance' in data
    assert 'Expected_value' in data

def test_prediction_route_invalid_id(client):
    # Testez la route de prédiction pour un ID client inexistant
    response = client.get('/prediction/999999')
    assert response.status_code == 200
    data = response.json
    assert 'Error' in data
    assert data['Error'] == 'Client ID not found'
