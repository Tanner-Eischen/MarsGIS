import os
import json
from fastapi.testclient import TestClient
from marshab.web.api import app


client = TestClient(app)


def test_demo_mode_read_write():
    r = client.get("/api/v1/demo-mode")
    assert r.status_code == 200
    data = r.json()
    assert "enabled" in data

    r2 = client.post("/api/v1/demo-mode", json={"enabled": True})
    assert r2.status_code == 200
    data2 = r2.json()
    assert data2["enabled"] is True

    r3 = client.post("/api/v1/demo-mode", json={"enabled": False})
    assert r3.status_code == 200
    data3 = r3.json()
    assert data3["enabled"] is False