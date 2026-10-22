from fastapi.testclient import TestClient
from backend.app.main import app
client=TestClient(app)
def test_health():assert client.get("/health").json()["status"]=="ok"
def test_invalid_age():assert client.post("/predict",json={"age":999}).status_code==422
def test_non_csv_rejected():assert client.post("/validate-dataset",files={"file":("a.txt",b"x,y","text/plain")}).status_code==400
