from fastapi.testclient import TestClient
from mine_safety import server
from mine_safety.server import app

client = TestClient(app)


def test_health_and_status_contract():
    assert client.get("/api/health").json()["status"] == "ok"
    data = client.get("/api/status").json()
    for field in ["summary", "metrics", "cameras", "incidents", "timeline", "sources", "assets", "limitations"]:
        assert field in data


def test_asset_does_not_expose_source_files():
    assert client.get("/api/asset/mine_safety/server.py").status_code == 404
    assert client.get("/api/asset/data/../../pyproject.toml").status_code in (404, 422)


def test_untrusted_input_is_not_run_as_a_path():
    response = client.post("/api/analyze", json={"input": "/etc/passwd"})
    assert response.status_code == 400


def test_download_has_provenance_label():
    response = client.get("/api/download/events")
    assert response.status_code == 200
    assert '"source_type":"synthetic"' in response.text


def test_private_report_is_not_served_even_if_present_locally(tmp_path, monkeypatch):
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "Mine_Safety_Project_Report.pdf").write_bytes(b"%PDF-private-notes")
    monkeypatch.setattr(server, "ROOT", tmp_path)

    assert client.get("/api/download/report").status_code == 404
    assert client.get("/api/asset/reports/Mine_Safety_Project_Report.pdf").status_code == 404
    assert "report_ready" not in client.get("/api/status").json()
