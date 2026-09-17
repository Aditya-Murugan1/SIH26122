import json

from backend.app import cpm_state, connect, cosine_similarity, embed_text, extract_event


def test_extraction_preserves_ocr_event_type_and_spans():
    event = extract_event("Spool X X - 1O2 started at 09:30 AM between rack 3 and V12.", "qa", 1)
    assert "XX102" in event["identifiers"]
    assert event["start_time"] == "09:30 AM"
    assert "R03" in event["location_terms"] and "V12" in event["location_terms"]
    assert event["source_spans"]["time"] == [27, 35]
    assert event["source_spans"]["identifier"]


def test_multiple_events_are_persisted():
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        response = client.post("/api/v1/reports", json={"text": "XX102 erection completed and cable tray CT103 installation started at 10:00.", "source": "qa"})
        assert response.status_code == 200
        body = response.json()
        assert len(body["events"]) == 2
        assert {item["event"]["event_type"] for item in body["events"]} == {"actual_finish", "actual_start"}


def test_auto_match_updates_actual_and_repeat_confirm_is_rejected():
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        response = client.post("/api/v1/reports", json={"text": "Spool XX102 completed at 4:45 PM.", "source": "qa"})
        body = response.json()
        assert body["match"]["decision"] == "AUTO_MATCHED"
        activity = client.get(f"/api/v1/activities/{body['match']['activity_id']}").json()
        assert activity["percent_complete"] == 100
        assert body["update"] is not None
        repeated = client.post(f"/api/v1/events/{body['event_id']}/confirm", json={"actor": "planner"})
        assert repeated.status_code == 409
        audit = client.get("/api/v1/audit").json()
        assert audit[0]["event_id"] == body["event_id"]
        assert "sub_scores" in json.loads(audit[0]["evidence"])


def test_import_materializes_dependencies_and_cpm_chain(tmp_path):
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        content = b"activity_code,description,wbs_path,level,discipline,location,planned_start,planned_finish,parent_activity\nA,Foundation,WBS,6,civil,A,2026-01-01,2026-01-02,\nB,Wall,WBS,6,civil,A,2026-01-01,2026-01-03,A\nC,Rooftop,WBS,6,civil,A,2026-01-01,2026-01-01,B\n"
        response = client.post("/api/v1/projects/1/schedule/import", files={"file": ("chain.csv", content, "text/csv")})
        assert response.status_code == 200
        assert response.json()["inserted"] == 3
        db = connect()
        edges = db.execute("SELECT COUNT(*) FROM activity_dependencies WHERE project_id=1").fetchone()[0]
        assert edges >= 2
        db.close()


def test_cpm_linear_chain_has_expected_float():
    db = connect()
    db.execute("INSERT INTO projects(id,name,created_at) VALUES (2,'CPM test','now')")
    for code, duration in (("A", 2), ("B", 3), ("C", 1)):
        db.execute("INSERT INTO activities(project_id,activity_code,description,wbs_path,level,discipline,planned_start,planned_finish,planned_duration) VALUES (2,?,?,?,?,?,?,?,?)", (code, code, "WBS", 6, "civil", "2026-01-01", "2026-01-02", duration))
    ids = [row[0] for row in db.execute("SELECT id FROM activities WHERE project_id=2 ORDER BY id")]
    db.execute("INSERT INTO activity_dependencies(project_id,predecessor_id,successor_id) VALUES (2,?,?)", (ids[0], ids[1]))
    db.execute("INSERT INTO activity_dependencies(project_id,predecessor_id,successor_id) VALUES (2,?,?)", (ids[1], ids[2]))
    db.commit()
    state = cpm_state(db, 2)
    assert state[ids[0]]["early_start"] == 0
    assert state[ids[0]]["early_finish"] == 2
    assert state[ids[2]]["early_finish"] == 6
    assert all(item["critical"] for item in state.values())
    db.close()


def test_real_minilm_embedding_handles_semantic_wording():
    report_vector = embed_text("XX102 spool installation completed")
    schedule_vector = embed_text("Erect Line 24-XX-102")
    assert report_vector and len(report_vector) == 384
    assert cosine_similarity(report_vector, schedule_vector) > 0.35


def test_time_agent_session_merges_answers_and_is_isolated():
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        first = client.post("/api/v1/agent/message", json={"text": "XX102 started", "session_id": "session-a"}).json()
        assert first["needs_clarification"] is True
        second = client.post("/api/v1/agent/message", json={"text": "Rack R03", "session_id": "session-a"}).json()
        assert second["needs_clarification"] is True and second["clarification"] == "time"
        final = client.post("/api/v1/agent/message", json={"text": "9:30 AM", "session_id": "session-a"}).json()
        assert final["event"]["identifiers"] == ["XX102"]
        assert final["event"]["location_terms"] == "R03"
        assert client.post("/api/v1/agent/message", json={"text": "9:30 AM", "session_id": "unknown"}).json()["event"]["identifiers"] == []


def test_memory_uses_actual_duration_not_planned_duration():
    from fastapi.testclient import TestClient
    from backend.app import app

    with TestClient(app) as client:
        report = client.post("/api/v1/reports", json={"text": "Spool XX102 started at 9:30 AM.", "source": "qa"}).json()
        client.post(f"/api/v1/events/{report['event_id']}/confirm", json={"actor": "planner"})
        report = client.post("/api/v1/reports", json={"text": "Spool XX102 completed at 4:45 PM.", "source": "qa"}).json()
        client.post(f"/api/v1/events/{report['event_id']}/confirm", json={"actor": "planner"})
        summary = client.get("/api/v1/memory/summary?description=piping%20erection").json()
        assert summary["count"] >= 1
        assert all(row["actual_duration"] is not None for row in summary["records"])


def test_cpm_parallel_branches_selects_longest_path():
    db = connect()
    db.execute("INSERT INTO projects(id,name,created_at) VALUES (2,'branch CPM','now')")
    for code, duration in (("A", 2), ("B", 5), ("C", 1), ("D", 2)):
        db.execute("INSERT INTO activities(project_id,activity_code,description,wbs_path,level,discipline,planned_start,planned_finish,planned_duration) VALUES (2,?,?,?,?,?,?,?,?)", (code, code, "WBS", 6, "civil", "2026-01-01", "2026-01-02", duration))
    ids = {row[1]: row[0] for row in db.execute("SELECT id,activity_code FROM activities WHERE project_id=2")}
    for predecessor, successor in (("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")):
        db.execute("INSERT INTO activity_dependencies(project_id,predecessor_id,successor_id) VALUES (2,?,?)", (ids[predecessor], ids[successor]))
    db.commit()
    state = cpm_state(db, 2)
    assert state[ids["D"]]["early_finish"] == 9
    assert state[ids["B"]]["critical"] and not state[ids["C"]]["critical"]
    assert state[ids["C"]]["total_float"] == 4
    db.close()
