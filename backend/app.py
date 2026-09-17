from __future__ import annotations

import csv
import io
import json
import os
import re
import sqlite3
import time
import unicodedata
from contextlib import asynccontextmanager
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.getenv("PROGRESSSYNC_DB", ROOT / "data" / "progresssync.db"))
TERM_PATH = ROOT / "data" / "terminology_map.v1.yaml"
FRONTEND = ROOT / "frontend"
VENDOR = ROOT / "vendor"
EMBEDDING_MODEL_NAME = os.getenv("PROGRESSSYNC_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
_embedding_model = None
_embedding_model_attempted = False
DISCIPLINES = {"civil", "piping", "static_equipment", "rotating_equipment", "electrical", "instrumentation", "hse"}
SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (id INTEGER PRIMARY KEY, name TEXT NOT NULL, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS activities (
 id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, activity_code TEXT NOT NULL, description TEXT NOT NULL,
 wbs_path TEXT, level INTEGER NOT NULL, discipline TEXT NOT NULL, location TEXT, planned_start TEXT,
 planned_finish TEXT, planned_duration REAL, actual_start TEXT, actual_finish TEXT, percent_complete REAL DEFAULT 0,
 status TEXT DEFAULT 'not_started', contractor TEXT, resource TEXT, embedding TEXT DEFAULT '[]', at_risk INTEGER DEFAULT 0,
 UNIQUE(project_id, activity_code)
);
CREATE TABLE IF NOT EXISTS activity_dependencies (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, predecessor_id INTEGER NOT NULL, successor_id INTEGER NOT NULL, dependency_type TEXT DEFAULT 'FS', lag REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS cpm_state (activity_id INTEGER PRIMARY KEY, early_start TEXT, early_finish TEXT, late_start TEXT, late_finish TEXT, total_float REAL, critical INTEGER, project_finish TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS schedule_snapshots (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, reason TEXT, created_at TEXT NOT NULL, project_finish TEXT, state_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS field_reports (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, source TEXT NOT NULL, raw_text TEXT NOT NULL, submitted_at TEXT NOT NULL, latency_ms REAL, status TEXT DEFAULT 'received');
CREATE TABLE IF NOT EXISTS execution_events (id INTEGER PRIMARY KEY, report_id INTEGER NOT NULL, event_type TEXT, discipline TEXT, activity_terms TEXT, identifiers TEXT, location_terms TEXT, quantity REAL, unit TEXT, progress REAL, event_timestamp TEXT, source_text TEXT, source_span TEXT, extraction_confidence REAL, status TEXT DEFAULT 'PENDING');
CREATE TABLE IF NOT EXISTS match_candidates (id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, activity_id INTEGER NOT NULL, score_id REAL, score_lexical REAL, score_semantic REAL, score_context REAL, temporal_factor REAL, fused_score REAL, rank INTEGER, evidence_json TEXT);
CREATE TABLE IF NOT EXISTS match_decisions (id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, decision TEXT NOT NULL, activity_id INTEGER, top_score REAL, second_score REAL, margin REAL, actor TEXT, comment TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS terminology_map (id INTEGER PRIMARY KEY, field_term TEXT, canonical_term TEXT, discipline TEXT, version TEXT, source TEXT);
CREATE TABLE IF NOT EXISTS delay_causes (id INTEGER PRIMARY KEY, event_id INTEGER NOT NULL, cause TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS audit_log (id INTEGER PRIMARY KEY, project_id INTEGER NOT NULL, activity_id INTEGER, event_id INTEGER, old_value TEXT, new_value TEXT, source TEXT, evidence TEXT, score REAL, actor TEXT, approval_status TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS agent_sessions (session_id TEXT PRIMARY KEY, last_report_id INTEGER, last_event_id INTEGER, pending_action TEXT, known_json TEXT DEFAULT '{}', unresolved_json TEXT DEFAULT '[]', updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS execution_memory (
 id INTEGER PRIMARY KEY, activity_id INTEGER NOT NULL, event_id INTEGER, activity_type TEXT, discipline TEXT, location TEXT,
 planned_duration REAL, actual_duration REAL, variance REAL, delay_cause TEXT, execution_notes TEXT, created_at TEXT NOT NULL
);
"""


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def embedding_model():
    global _embedding_model, _embedding_model_attempted
    if _embedding_model_attempted:
        return _embedding_model
    _embedding_model_attempted = True
    try:
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, local_files_only=True)
    except Exception:
        _embedding_model = None
    return _embedding_model


def embed_text(text: str) -> list[float]:
    model = embedding_model()
    if model is None:
        return []
    return [round(float(value), 8) for value in model.encode([text], normalize_embeddings=True)[0]]


def cosine_similarity(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left_array, right_array = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    return float(np.dot(left_array, right_array) / denominator) if denominator else 0.0


def populate_activity_embeddings(db: sqlite3.Connection, project_id: int) -> None:
    rows = db.execute("SELECT id, description FROM activities WHERE project_id=?", (project_id,)).fetchall()
    for row in rows:
        if not row["description"]:
            continue
        vector = embed_text(row["description"])
        if vector:
            db.execute("UPDATE activities SET embedding=? WHERE id=?", (json.dumps(vector), row["id"]))


def init_db() -> None:
    with closing(connect()) as db:
        db.executescript(SCHEMA)
        for column, definition in (("start_time", "TEXT"), ("end_time", "TEXT"), ("evidence_json", "TEXT")):
            try:
                db.execute(f"ALTER TABLE execution_events ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError:
                pass
        for column, definition in (("known_json", "TEXT DEFAULT '{}'"), ("unresolved_json", "TEXT DEFAULT '[]'")):
            try:
                db.execute(f"ALTER TABLE agent_sessions ADD COLUMN {column} {definition}")
            except sqlite3.OperationalError:
                pass
        try:
            db.execute("ALTER TABLE execution_events ADD COLUMN embedding TEXT DEFAULT '[]'")
        except sqlite3.OperationalError:
            pass
        db.commit()
        if not db.execute("SELECT 1 FROM projects LIMIT 1").fetchone():
            db.execute("INSERT INTO projects(name, created_at) VALUES (?, ?)", ("OIL Demo Project", now()))
            load_terms(db)
            seed_schedule(db)
            populate_activity_embeddings(db, 1)
            recompute(db, 1, "initial baseline")
        else:
            populate_activity_embeddings(db, 1)
            db.commit()


def load_terms(db: sqlite3.Connection) -> None:
    terms = []
    current = {}
    if TERM_PATH.exists():
        for line in TERM_PATH.read_text(encoding="utf-8").splitlines():
            match = re.match(r"\s*-?\s*(field_term|canonical_term|discipline):\s*(.+?)\s*$", line)
            if match:
                current[match.group(1)] = match.group(2).strip().strip("'\"")
                if len(current) == 3:
                    terms.append((current["field_term"], current["canonical_term"], current["discipline"]))
                    current = {}
    if not terms:
        terms = [("spool", "piping segment", "piping"), ("install", "erect", "civil"), ("erection", "erect", "static_equipment"), ("rack 3", "R03", "piping"), ("rack-3", "R03", "piping"), ("CT103", "CT-103", "rotating_equipment"), ("pump", "pump", "rotating_equipment"), ("boltup", "bolting", "piping"), ("hydrotest", "pressure test", "piping"), ("cable pull", "cable installation", "electrical")]
    db.executemany("INSERT INTO terminology_map(field_term, canonical_term, discipline, version, source) VALUES (?, ?, ?, 'v1', 'OIL field lexicon')", terms)


def seed_schedule(db: sqlite3.Connection) -> None:
    rows = []
    base = date.today() - timedelta(days=12)
    templates = [
        ("CIV", "Foundation and concrete works", "civil", "Area A"),
        ("PIP", "Erect line XX102 spool", "piping", "R03"),
        ("STA", "Erect vessel V-201", "static_equipment", "Unit 2"),
        ("ROT", "Install pump CT-103", "rotating_equipment", "Unit 2"),
        ("ELE", "Install cable tray", "electrical", "R03"),
        ("INS", "Calibrate pressure transmitter", "instrumentation", "Unit 2"),
        ("HSE", "Permit and safety inspection", "hse", "Site"),
    ]
    for i in range(35):
        prefix, desc, discipline, location = templates[i % len(templates)]
        code = f"{prefix}-L5-{i+1:03d}"
        if discipline == "piping":
            desc = f"Erect line XX{102 + i // 7:03d} spool"
        start = base + timedelta(days=i % 9)
        finish = start + timedelta(days=2 + i % 4)
        rows.append((1, code, f"{desc} {i+1:03d}", f"{discipline.upper()} / {location} / PACKAGE {i//7+1}", 5 if i % 4 else 6, discipline, location, start.isoformat(), finish.isoformat(), (finish-start).days + 1, 0, "not_started"))
    db.executemany("INSERT INTO activities(project_id,activity_code,description,wbs_path,level,discipline,location,planned_start,planned_finish,planned_duration,percent_complete,status) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    activities = db.execute("SELECT id FROM activities WHERE project_id=1 ORDER BY id").fetchall()
    edges = [(1, activities[i-1][0], activities[i][0], "FS", 0) for i in range(1, len(activities))]
    db.executemany("INSERT INTO activity_dependencies(project_id,predecessor_id,successor_id,dependency_type,lag) VALUES (?,?,?,?,?)", edges)


def parse_day(value: Any) -> date | None:
    if not value:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%b-%Y", "%Y/%m/%d"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            pass
    return None


def norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode().lower()
    text = re.sub(r"([a-z])\s+([a-z])\s*[- ]\s*([0-9o])", r"\1\2\3", text)
    text = text.replace("1o", "10")
    text = re.sub(r"([a-z])([0-9])", r"\1 \2", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> set[str]:
    return set(norm(text).split())


def extract_event(text: str, source: str, report_id: int) -> dict[str, Any]:
    clean = text.strip()
    normalized = norm(clean)
    discipline = next((d for d in DISCIPLINES if d.replace("_", " ") in normalized or d.split("_")[0] in normalized), None)
    if "pipe" in normalized or "spool" in normalized or "hydro" in normalized or "erection" in normalized or "erect" in normalized:
        discipline = "piping"
    elif "pump" in normalized or "rotating" in normalized:
        discipline = "rotating_equipment"
    elif "vessel" in normalized or "static" in normalized:
        discipline = "static_equipment"
    elif "cable" in normalized or "electrical" in normalized:
        discipline = "electrical"
    elif "transmitter" in normalized or "instrument" in normalized:
        discipline = "instrumentation"
    location_values = []
    rack = re.search(r"\b(?:rack|r)[ -]?(\d+)\b", clean, re.I)
    if rack:
        location_values.append(f"R{int(rack.group(1)):02d}")
    location_values.extend(match.group(0).upper() for match in re.finditer(r"\bV\d{1,4}\b", clean, re.I))
    location = "; ".join(dict.fromkeys(location_values)) or None
    equipment_matches = list(re.finditer(r"\b(?:[A-Z]{2,5}(?:[- ]?[A-Z0-9]{1,5}){1,3}|[A-Z]{1,3}\d{3,4})\b", clean))
    equipment = [match.group(0) for match in equipment_matches]
    ocr_identifier = re.search(r"\bX\s*X\s*[- ]?\s*1[O0]2\b", clean, re.I)
    if ocr_identifier and "XX102" not in [norm(value).replace(" ", "") for value in equipment]:
        equipment.append("XX102")
    equipment = list(dict.fromkeys(equipment))
    progress_match = re.search(r"(\d{1,3})\s*%", clean)
    quantity_match = re.search(r"(?:qty|quantity|installed|completed)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(m|meter|meters|mm|inch|in|ea|each)?", clean, re.I)
    time_match = re.search(r"(?:at|@)\s*(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)", clean, re.I)
    if not time_match:
        time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:am|pm)?)\b", clean, re.I)
    if not time_match:
        time_match = re.search(r"\baround\s+(\d{1,2})\b", clean, re.I)
    time_text = time_match.group(1) if time_match else None
    event_type = "progress_update"
    if re.search(r"\b(started|commenced|began)\b", normalized):
        event_type = "actual_start"
    elif re.search(r"\b(finished|complete|completed|ended)\b", normalized):
        event_type = "actual_finish"
    time_span = [time_match.start(1), time_match.end(1)] if time_match else None
    identifier_span = [equipment_matches[0].start(), equipment_matches[0].end()] if equipment_matches else ([ocr_identifier.start(), ocr_identifier.end()] if ocr_identifier else None)
    source_spans = {"report": [0, len(clean)], "time": time_span, "identifier": identifier_span, "location": [rack.start(), rack.end()] if rack else None}
    activity_terms = clean
    span = [0, len(clean)]
    event_timestamp = (date.today() - timedelta(days=1)).isoformat() if re.search(r"\byesterday\b", normalized) else date.today().isoformat()
    progress = float(progress_match.group(1)) if progress_match else (100.0 if event_type == "actual_finish" else (0.0 if event_type == "actual_start" else None))
    return {"report_id": report_id, "event_type": event_type, "discipline": discipline, "activity_terms": activity_terms, "identifiers": equipment, "location_terms": location, "quantity": float(quantity_match.group(1)) if quantity_match else None, "unit": quantity_match.group(2) if quantity_match else None, "progress": progress, "event_timestamp": event_timestamp, "start_time": time_text if event_type == "actual_start" else None, "end_time": time_text if event_type == "actual_finish" else None, "source_text": clean, "source_span": span, "source_spans": source_spans, "extraction_confidence": 0.86 if discipline else 0.62, "time_text": time_text}


def similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, norm(a), norm(b)).ratio()


def score_candidate(event: dict[str, Any], activity: sqlite3.Row, terms: list[sqlite3.Row]) -> tuple[dict[str, float], dict[str, Any]]:
    field = norm(event["activity_terms"])
    description = norm(activity["description"])
    activity_code = norm(activity["activity_code"])
    score_id = 0.0
    if any(norm(identifier) in activity_code or norm(identifier) in description for identifier in event["identifiers"]):
        score_id = 1.0
    mapped = []
    mapped_field = field
    for term in terms:
        if term["field_term"] in mapped_field:
            mapped.append(dict(term))
            mapped_field = mapped_field.replace(term["field_term"], term["canonical_term"])
    overlap = len(tokens(mapped_field) & tokens(description)) / max(1, len(tokens(description)))
    score_lexical = max(similarity(mapped_field, description), similarity(field, description), overlap)
    if score_id >= 1.0:
        score_lexical = max(score_lexical, 0.9)
    event_embedding = event.get("embedding") or embed_text(event["activity_terms"])
    raw_activity_embedding = activity["embedding"] if hasattr(activity, "keys") and "embedding" in activity.keys() else "[]"
    activity_embedding = json.loads(raw_activity_embedding or "[]") if raw_activity_embedding else []
    semantic_cosine = cosine_similarity(event_embedding, activity_embedding)
    score_semantic = semantic_cosine if semantic_cosine is not None else min(1.0, score_lexical * 0.88 + (0.12 if any(t in mapped_field for t in tokens(description)) else 0))
    context = 0.0
    if event.get("discipline") and event["discipline"] == activity["discipline"]:
        context += 0.6
    elif event.get("discipline") and event["discipline"] != activity["discipline"]:
        context -= 0.35
    if event.get("location_terms") and norm(activity["location"]) in norm(event["location_terms"]):
        context += 0.4
    score_context = max(0.0, min(1.0, 0.5 + context))
    temporal_factor = 1.0
    event_day = parse_day(event.get("event_timestamp"))
    start, finish = parse_day(activity["planned_start"]), parse_day(activity["planned_finish"])
    if event_day and start and finish:
        if start <= event_day <= finish:
            temporal_factor = 1.0
        elif score_id >= 1.0:
            temporal_factor = 1.0
        elif event_day < start:
            temporal_factor = 0.94
        else:
            temporal_factor = 0.9
    base = 0.30*score_id + 0.15*score_lexical + 0.25*score_semantic + 0.30*score_context
    evidence = {"activity_code": activity["activity_code"], "description": activity["description"], "mapped_terms": mapped, "source_span": event["source_span"], "semantic_model": EMBEDDING_MODEL_NAME if semantic_cosine is not None else "deterministic-fallback"}
    return {"score_id": score_id, "score_lexical": score_lexical, "score_semantic": score_semantic, "score_context": score_context, "temporal_factor": temporal_factor, "fused_score": base*temporal_factor}, evidence


def make_match(db: sqlite3.Connection, event_id: int, event: dict[str, Any], project_id: int = 1) -> dict[str, Any]:
    activities = db.execute("SELECT * FROM activities WHERE project_id=? ORDER BY id", (project_id,)).fetchall()
    terms = db.execute("SELECT * FROM terminology_map").fetchall()
    candidates = []
    for activity in activities:
        if event.get("discipline") and event["discipline"] != activity["discipline"] and event.get("extraction_confidence", 0) >= 0.8:
            continue
        scores, evidence = score_candidate(event, activity, terms)
        candidates.append((scores, evidence, activity))
    if not candidates:
        candidates = [(score_candidate(event, event, terms)[0], {}, event)]
    candidates.sort(key=lambda item: item[0]["fused_score"], reverse=True)
    for rank, (scores, evidence, activity) in enumerate(candidates[:20], 1):
        db.execute("INSERT INTO match_candidates(event_id,activity_id,score_id,score_lexical,score_semantic,score_context,temporal_factor,fused_score,rank,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?)", (event_id, activity["id"], scores["score_id"], scores["score_lexical"], scores["score_semantic"], scores["score_context"], scores["temporal_factor"], scores["fused_score"], rank, json.dumps(evidence)))
    top = candidates[0][0]["fused_score"]
    second = candidates[1][0]["fused_score"] if len(candidates) > 1 else 0
    margin = top - second
    weak_unidentified = candidates[0][0]["score_id"] == 0 and candidates[0][0]["score_lexical"] < 0.30 and not event.get("discipline") and not event.get("location_terms")
    decision = "UNMATCHED" if weak_unidentified or top < 0.45 else "AUTO_MATCHED" if top >= 0.82 and margin >= 0.12 else "REVIEW_REQUIRED"
    activity_id = candidates[0][2]["id"] if decision != "UNMATCHED" else None
    db.execute("INSERT INTO match_decisions(event_id,decision,activity_id,top_score,second_score,margin,actor,created_at) VALUES (?,?,?,?,?,?,?,?)", (event_id, decision, activity_id, top, second, margin, "system", now()))
    db.execute("UPDATE execution_events SET status=? WHERE id=?", (decision, event_id))
    db.commit()
    return {"decision": decision, "top_score": top, "second_score": second, "margin": margin, "activity_id": activity_id}


def cpm_state(db: sqlite3.Connection, project_id: int) -> dict[int, dict[str, Any]]:
    acts = db.execute("SELECT * FROM activities WHERE project_id=?", (project_id,)).fetchall()
    edges = db.execute("SELECT predecessor_id, successor_id, lag FROM activity_dependencies WHERE project_id=?", (project_id,)).fetchall()
    graph = nx.DiGraph()
    graph.add_nodes_from(a["id"] for a in acts)
    graph.add_edges_from((e["predecessor_id"], e["successor_id"], {"lag": e["lag"]}) for e in edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("Dependency graph contains a cycle")
    by_id = {a["id"]: a for a in acts}
    def effective_duration(activity: sqlite3.Row) -> float:
        planned = float(activity["planned_duration"] or 1)
        if activity["actual_start"] and activity["actual_finish"]:
            start = parse_day(activity["actual_start"])
            finish = parse_day(activity["actual_finish"])
            if start and finish:
                return max(planned, float((finish - start).days + 1))
        if activity["actual_finish"] and activity["planned_finish"]:
            actual_finish = parse_day(activity["actual_finish"])
            planned_finish = parse_day(activity["planned_finish"])
            if actual_finish and planned_finish and actual_finish > planned_finish:
                return planned + float((actual_finish - planned_finish).days)
        return planned
    es: dict[int, float] = {}
    ef: dict[int, float] = {}
    for node in nx.topological_sort(graph):
        duration = effective_duration(by_id[node])
        es[node] = max((ef[p] + float(graph[p][node].get("lag", 0)) for p in graph.predecessors(node)), default=0)
        ef[node] = es[node] + duration
    project_finish = max(ef.values(), default=0)
    ls: dict[int, float] = {}
    lf: dict[int, float] = {}
    for node in reversed(list(nx.topological_sort(graph))):
        duration = effective_duration(by_id[node])
        lf[node] = min((ls[s] - float(graph[node][s].get("lag", 0)) for s in graph.successors(node)), default=project_finish)
        ls[node] = lf[node] - duration
    result = {}
    for node in graph.nodes:
        result[node] = {"early_start": es[node], "early_finish": ef[node], "late_start": ls[node], "late_finish": lf[node], "total_float": round(ls[node]-es[node], 2), "critical": abs(ls[node]-es[node]) < 0.001, "project_finish": project_finish}
    return result


def recompute(db: sqlite3.Connection, project_id: int, reason: str) -> dict[str, Any]:
    state = cpm_state(db, project_id)
    serialized = json.dumps(state)
    finish = max((v["project_finish"] for v in state.values()), default=0)
    db.execute("INSERT INTO schedule_snapshots(project_id,reason,created_at,project_finish,state_json) VALUES (?,?,?,?,?)", (project_id, reason, now(), finish, serialized))
    for activity_id, values in state.items():
        db.execute("INSERT OR REPLACE INTO cpm_state(activity_id,early_start,early_finish,late_start,late_finish,total_float,critical,project_finish,updated_at) VALUES (?,?,?,?,?,?,?,?,?)", (activity_id, values["early_start"], values["early_finish"], values["late_start"], values["late_finish"], values["total_float"], int(values["critical"]), finish, now()))
        db.execute("UPDATE activities SET at_risk=? WHERE id=?", (int(values["total_float"] <= 1 and not values["critical"]), activity_id))
    db.commit()
    return {"project_finish": finish, "critical": [k for k, v in state.items() if v["critical"]], "state": state}


def apply_update(db: sqlite3.Connection, project_id: int, event_id: int, activity_id: int, actor: str, comment: str = "") -> dict[str, Any]:
    event = db.execute("SELECT * FROM execution_events WHERE id=?", (event_id,)).fetchone()
    decision = db.execute("SELECT * FROM match_decisions WHERE event_id=? ORDER BY id DESC LIMIT 1", (event_id,)).fetchone()
    activity = db.execute("SELECT * FROM activities WHERE id=? AND project_id=?", (activity_id, project_id)).fetchone()
    if not event or not decision or not activity:
        raise HTTPException(404, "Event, decision, or activity not found")
    if decision["decision"] in {"APPROVED", "REJECTED"}:
        raise HTTPException(409, "This event already has a final planner decision")
    before = recompute(db, project_id, "pre-update snapshot")
    old = {"percent_complete": activity["percent_complete"], "actual_start": activity["actual_start"], "actual_finish": activity["actual_finish"], "status": activity["status"]}
    progress = event["progress"] if event["progress"] is not None else old["percent_complete"]
    status = "complete" if progress >= 100 else "in_progress" if progress > 0 or event["event_type"] == "actual_start" else old["status"]
    actual_start = old["actual_start"] or event["event_timestamp"] if progress > 0 or event["event_type"] == "actual_start" else old["actual_start"]
    actual_finish = event["event_timestamp"] if progress >= 100 or event["event_type"] == "actual_finish" else old["actual_finish"]
    new = {"percent_complete": progress, "actual_start": actual_start, "actual_finish": actual_finish, "status": status}
    db.execute("UPDATE activities SET percent_complete=?,actual_start=?,actual_finish=?,status=? WHERE id=?", (progress, actual_start, actual_finish, status, activity_id))
    db.execute("UPDATE match_decisions SET decision=?,activity_id=?,actor=?,comment=? WHERE id=?", ("APPROVED", activity_id, actor, comment, decision["id"]))
    candidate = db.execute("SELECT * FROM match_candidates WHERE event_id=? AND activity_id=? ORDER BY rank LIMIT 1", (event_id, activity_id)).fetchone()
    evidence = json.dumps({"source_text": event["source_text"], "source_spans": json.loads(event["evidence_json"] or "{}") if event["evidence_json"] else {}, "decision": decision["decision"], "sub_scores": dict(candidate) if candidate else {}})
    db.execute("INSERT INTO audit_log(project_id,activity_id,event_id,old_value,new_value,source,evidence,score,actor,approval_status,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (project_id, activity_id, event_id, json.dumps(old), json.dumps(new), "field_report", evidence, decision["top_score"] if decision else None, actor, "approved", now()))
    actual_duration = None
    if new["actual_start"] and new["actual_finish"]:
        actual_start_day, actual_finish_day = parse_day(new["actual_start"]), parse_day(new["actual_finish"])
        if actual_start_day and actual_finish_day:
            actual_duration = max(0, (actual_finish_day - actual_start_day).days + 1)
    delay = db.execute("SELECT cause FROM delay_causes WHERE event_id=? ORDER BY id DESC LIMIT 1", (event_id,)).fetchone()
    db.execute("INSERT INTO execution_memory(activity_id,event_id,activity_type,discipline,location,planned_duration,actual_duration,variance,delay_cause,execution_notes,created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)", (activity_id, event_id, activity["description"], activity["discipline"], activity["location"], activity["planned_duration"], actual_duration, actual_duration - activity["planned_duration"] if actual_duration is not None else None, delay["cause"] if delay else None, event["source_text"], now()))
    after = recompute(db, project_id, "confirmed actual update")
    db.commit()
    return {"old": old, "new": new, "before": before, "after": after, "critical_path_changed": before["critical"] != after["critical"]}


class ReportIn(BaseModel):
    project_id: int = 1
    text: str = Field(min_length=1, max_length=20000)
    source: str = "text"


class DecisionIn(BaseModel):
    activity_id: int | None = None
    actor: str = "planner"
    comment: str = ""


class AgentIn(BaseModel):
    text: str = Field(min_length=1, max_length=20000)
    session_id: str | None = None


def save_agent_session(db: sqlite3.Connection, session_id: str, report_id: int | None, event_id: int | None, pending_action: str | None, known: dict[str, Any], unresolved: list[str]) -> None:
    db.execute("INSERT OR REPLACE INTO agent_sessions(session_id,last_report_id,last_event_id,pending_action,known_json,unresolved_json,updated_at) VALUES (?,?,?,?,?,?,?)", (session_id, report_id, event_id, pending_action, json.dumps(known), json.dumps(unresolved), now()))


def agent_clarification(session_id: str, text: str) -> dict[str, Any] | None:
    with closing(connect()) as db:
        session = db.execute("SELECT * FROM agent_sessions WHERE session_id=?", (session_id,)).fetchone()
        if not session:
            return None
        known = json.loads(session["known_json"] or "{}")
        combined = f"{known.get('raw_text', '')} {text}".strip()
        event = extract_event(combined, "time_agent", session["last_report_id"] or 0)
        missing = []
        if not event.get("identifiers"):
            missing.append("activity")
        if not event.get("time_text"):
            missing.append("time")
        if missing:
            save_agent_session(db, session_id, session["last_report_id"], session["last_event_id"], missing[0], {"raw_text": combined}, missing)
            db.commit()
            question = "Which activity or location do you mean?" if "activity" in missing else "What time did that happen? Include a time such as 09:30 AM."
            return {"reply": question, "needs_clarification": True, "clarification": missing[0], "session_id": session_id}
        db.execute("DELETE FROM agent_sessions WHERE session_id=?", (session_id,))
        db.commit()
        return report(ReportIn(text=combined, source="time_agent"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    init_db()
    yield


app = FastAPI(title="ProgressSync AI", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.get("/")
def index() -> FileResponse:
    return FileResponse(FRONTEND / "index.html")

@app.get("/app.js")
def js() -> FileResponse:
    return FileResponse(FRONTEND / "app.js")

@app.get("/styles.css")
def css() -> FileResponse:
    return FileResponse(FRONTEND / "styles.css")

@app.get("/vendor/frappe-gantt.js")
def gantt_js() -> FileResponse:
    return FileResponse(VENDOR / "frappe-gantt.umd.js")

@app.get("/vendor/frappe-gantt.css")
def gantt_css() -> FileResponse:
    return FileResponse(VENDOR / "frappe-gantt.css")

@app.get("/api/v1/projects/{project_id}/activities")
def activities(project_id: int):
    with closing(connect()) as db:
        return [dict(row) for row in db.execute("SELECT a.*, c.total_float, c.critical FROM activities a LEFT JOIN cpm_state c ON a.id=c.activity_id WHERE a.project_id=? ORDER BY a.id", (project_id,))]

@app.get("/api/v1/activities/{activity_id}")
def activity(activity_id: int):
    with closing(connect()) as db:
        row = db.execute("SELECT a.*, c.* FROM activities a LEFT JOIN cpm_state c ON a.id=c.activity_id WHERE a.id=?", (activity_id,)).fetchone()
        if not row: raise HTTPException(404, "Activity not found")
        return dict(row)

@app.post("/api/v1/reports")
def report(payload: ReportIn):
    started = time.perf_counter()
    with closing(connect()) as db:
        report_id = db.execute("INSERT INTO field_reports(project_id,source,raw_text,submitted_at) VALUES (?,?,?,?)", (payload.project_id, payload.source, payload.text, now())).lastrowid
        chunks = re.split(r"\s+and\s+(?=(?:cable|spool|pump|vessel|foundation|line|xx\d))", payload.text, flags=re.I)
        results = []
        for chunk in chunks:
            event = extract_event(chunk, payload.source, report_id)
            event_id = db.execute("INSERT INTO execution_events(report_id,event_type,discipline,activity_terms,identifiers,location_terms,quantity,unit,progress,event_timestamp,source_text,source_span,extraction_confidence,start_time,end_time,evidence_json) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (report_id, event["event_type"], event["discipline"], event["activity_terms"], json.dumps(event["identifiers"]), event["location_terms"], event["quantity"], event["unit"], event["progress"], event["event_timestamp"], event["source_text"], json.dumps(event["source_span"]), event["extraction_confidence"], event["start_time"], event["end_time"], json.dumps(event["source_spans"]))).lastrowid
            event["id"] = event_id
            event["embedding"] = embed_text(event["activity_terms"])
            if event["embedding"]:
                db.execute("UPDATE execution_events SET embedding=? WHERE id=?", (json.dumps(event["embedding"]), event_id))
            match = make_match(db, event_id, event, payload.project_id)
            cause_match = re.search(r"(?:delay(?:ed)?|hold|blocked|late|weather|material|permit|access)\s+(?:due to|because of|for)?\s*([a-z][a-z ]{2,30})", chunk, re.I)
            if cause_match:
                db.execute("INSERT INTO delay_causes(event_id,cause,notes) VALUES (?,?,?)", (event_id, cause_match.group(1).strip().lower(), chunk))
            update = None
            if match["decision"] == "AUTO_MATCHED":
                update = apply_update(db, payload.project_id, event_id, match["activity_id"], "system")
            results.append({"event": event, "match": match, "update": update})
        latency = round((time.perf_counter() - started) * 1000, 2)
        status = "MULTI_EVENT" if len(results) > 1 else results[0]["match"]["decision"]
        db.execute("UPDATE field_reports SET latency_ms=?,status=? WHERE id=?", (latency, status, report_id))
        db.commit()
        first = results[0]
        return {"report_id": report_id, "event_id": first["event"]["id"], "event": first["event"], "match": first["match"], "update": first["update"], "events": results, "latency_ms": latency}

@app.get("/api/v1/events/{event_id}")
def event(event_id: int):
    with closing(connect()) as db:
        row = db.execute("SELECT * FROM execution_events WHERE id=?", (event_id,)).fetchone()
        if not row: raise HTTPException(404, "Event not found")
        data = dict(row); data["identifiers"] = json.loads(data["identifiers"] or "[]"); data["source_span"] = json.loads(data["source_span"] or "[]"); data["evidence"] = json.loads(data.get("evidence_json") or "{}")
        data["decision"] = dict(db.execute("SELECT * FROM match_decisions WHERE event_id=? ORDER BY id DESC LIMIT 1", (event_id,)).fetchone() or {})
        return data

@app.get("/api/v1/events/{event_id}/candidates")
def candidates(event_id: int):
    with closing(connect()) as db:
        rows = db.execute("SELECT c.*, a.activity_code, a.description, a.discipline, a.location, a.wbs_path FROM match_candidates c JOIN activities a ON a.id=c.activity_id WHERE c.event_id=? ORDER BY c.rank", (event_id,)).fetchall()
        result = [dict(row) for row in rows]
        for item in result: item["evidence"] = json.loads(item.pop("evidence_json") or "{}")
        return result

@app.get("/api/v1/review-queue")
def review_queue():
    with closing(connect()) as db:
        return [dict(row) for row in db.execute("SELECT e.*, d.decision, d.top_score, d.second_score, d.margin, d.activity_id FROM execution_events e JOIN match_decisions d ON d.event_id=e.id WHERE d.decision IN ('REVIEW_REQUIRED','UNMATCHED') ORDER BY e.id DESC")]

@app.post("/api/v1/events/{event_id}/confirm")
def confirm(event_id: int, payload: DecisionIn):
    with closing(connect()) as db:
        decision = db.execute("SELECT * FROM match_decisions WHERE event_id=? ORDER BY id DESC LIMIT 1", (event_id,)).fetchone()
        report_row = db.execute("SELECT project_id FROM field_reports WHERE id=(SELECT report_id FROM execution_events WHERE id=?)", (event_id,)).fetchone()
        project_id = report_row["project_id"] if report_row else 1
        activity_id = payload.activity_id or (decision["activity_id"] if decision else None)
        if not activity_id: raise HTTPException(400, "activity_id is required for an unmatched event")
        return apply_update(db, project_id, event_id, activity_id, payload.actor, payload.comment)

@app.post("/api/v1/events/{event_id}/reject")
def reject(event_id: int, payload: DecisionIn):
    with closing(connect()) as db:
        db.execute("UPDATE match_decisions SET decision='REJECTED',actor=?,comment=? WHERE event_id=?", (payload.actor, payload.comment, event_id)); db.execute("UPDATE execution_events SET status='REJECTED' WHERE id=?", (event_id,)); db.commit(); return {"event_id": event_id, "decision": "REJECTED"}

@app.post("/api/v1/projects/{project_id}/recompute")
def recompute_api(project_id: int):
    with closing(connect()) as db: return recompute(db, project_id, "manual recompute")

@app.get("/api/v1/projects/{project_id}/critical-path")
def critical_path(project_id: int):
    with closing(connect()) as db:
        return [dict(row) for row in db.execute("SELECT a.*, c.total_float, c.critical FROM activities a JOIN cpm_state c ON a.id=c.activity_id WHERE a.project_id=? AND c.critical=1 ORDER BY c.early_start", (project_id,))]

@app.get("/api/v1/projects/{project_id}/snapshots/{snapshot_id}/diff")
def snapshot_diff(project_id: int, snapshot_id: int):
    with closing(connect()) as db:
        current = db.execute("SELECT * FROM schedule_snapshots WHERE project_id=? ORDER BY id DESC LIMIT 1", (project_id,)).fetchone()
        old = db.execute("SELECT * FROM schedule_snapshots WHERE project_id=? AND id=?", (project_id, snapshot_id)).fetchone()
        if not old or not current: raise HTTPException(404, "Snapshot not found")
        before, after = json.loads(old["state_json"]), json.loads(current["state_json"])
        return {"from": old["id"], "to": current["id"], "project_finish_change": current["project_finish"] - old["project_finish"], "critical_entered": [k for k in after if after[k]["critical"] and not before.get(k, {}).get("critical")], "critical_left": [k for k in before if before[k]["critical"] and not after.get(k, {}).get("critical")], "float_warnings": [k for k, value in after.items() if value["total_float"] <= 1]}

@app.get("/api/v1/analytics/discipline-productivity")
def productivity():
    with closing(connect()) as db:
        return [dict(row) for row in db.execute("SELECT discipline, COUNT(*) activities, ROUND(AVG(percent_complete),1) progress, ROUND(AVG(CASE WHEN actual_finish IS NOT NULL THEN planned_duration END),1) actual_duration, ROUND(AVG(CASE WHEN actual_finish IS NOT NULL THEN percent_complete END),1) completed_progress FROM activities GROUP BY discipline ORDER BY discipline")]

@app.get("/api/v1/analytics/delay-causes")
def delay_causes():
    with closing(connect()) as db: return [dict(row) for row in db.execute("SELECT cause, COUNT(*) count FROM delay_causes GROUP BY cause ORDER BY count DESC")]

@app.get("/api/v1/analytics/variance")
def variance():
    with closing(connect()) as db: return [dict(row) for row in db.execute("SELECT wbs_path, ROUND(AVG(percent_complete),1) progress, COUNT(*) activities FROM activities GROUP BY wbs_path ORDER BY wbs_path")]

@app.get("/api/v1/memory/similar")
def memory(description: str = "piping erection"):
    with closing(connect()) as db:
        rows = db.execute("SELECT m.*, a.activity_code FROM execution_memory m JOIN activities a ON a.id=m.activity_id WHERE m.actual_duration IS NOT NULL ORDER BY m.id DESC").fetchall()
        matches = [{"activity_code": r["activity_code"], "description": r["activity_type"], "discipline": r["discipline"], "location": r["location"], "actual_duration": r["actual_duration"], "planned_duration": r["planned_duration"], "variance": r["variance"], "delay_cause": r["delay_cause"], "execution_notes": r["execution_notes"], "similarity": similarity(description, r["activity_type"])} for r in rows]
        return sorted(matches, key=lambda item: item["similarity"], reverse=True)[:10]


@app.get("/api/v1/memory/summary")
def memory_summary(description: str = "piping erection"):
    matches = memory(description)
    durations = [row["actual_duration"] for row in matches if row["actual_duration"] is not None]
    return {"query": description, "count": len(durations), "average_actual_duration": round(sum(durations) / len(durations), 2) if durations else None, "records": matches}

@app.get("/api/v1/audit")
def audit():
    with closing(connect()) as db: return [dict(row) for row in db.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT 100")]

@app.get("/api/v1/terminology-map")
def terminology():
    with closing(connect()) as db: return [dict(row) for row in db.execute("SELECT * FROM terminology_map ORDER BY field_term")]

@app.post("/api/v1/agent/message")
def agent(payload: AgentIn):
    if payload.session_id:
        continuation = agent_clarification(payload.session_id, payload.text)
        if continuation:
            if "reply" not in continuation:
                continuation["reply"] = "I found a schedule event through the same pipeline."
            return continuation
    if not payload.session_id and not re.search(r"\b(?:at|@)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b|\b\d{1,2}:\d{2}\b", payload.text, re.I) and re.search(r"\b(started|finished|completed)\b", payload.text, re.I):
        return {"reply": "What time did that happen? Include a time such as 09:30 AM so I can preserve the field evidence.", "needs_clarification": True, "clarification": "event_time", "session_id": payload.session_id}
    if payload.session_id and re.search(r"\b(started|finished|completed)\b", payload.text, re.I) and not re.search(r"\b(?:at|@)\s*\d{1,2}(?::\d{2})?\s*(?:am|pm)?\b|\b\d{1,2}:\d{2}\b", payload.text, re.I):
        with closing(connect()) as db:
            event = extract_event(payload.text, "time_agent", 0)
            save_agent_session(db, payload.session_id, None, None, "time", {"raw_text": payload.text, "identifiers": event["identifiers"]}, ["time"])
            db.commit()
        return {"reply": "What time did that happen? Include a time such as 09:30 AM to preserve the field evidence.", "needs_clarification": True, "clarification": "time", "session_id": payload.session_id}
    result = report(ReportIn(text=payload.text, source="time_agent"))
    decision = result["match"]["decision"]
    reply = "I found a confident schedule activity. Please confirm." if decision == "AUTO_MATCHED" else "I found multiple plausible activities. Please review the evidence." if decision == "REVIEW_REQUIRED" else "I could not find a schedule counterpart, so I preserved the observation for review."
    result["reply"] = reply
    result["needs_clarification"] = decision == "REVIEW_REQUIRED"
    return result

@app.post("/api/v1/projects/{project_id}/schedule/import")
async def import_schedule(project_id: int, file: UploadFile = File(...)):
    content = await file.read()
    if len(content) > 10_000_000: raise HTTPException(413, "File exceeds 10 MB limit")
    name = (file.filename or "").lower()
    if not (name.endswith(".csv") or name.endswith(".xlsx")): raise HTTPException(415, "Only CSV and XLSX schedules are supported")
    if name.endswith(".xlsx"):
        try:
            from openpyxl import load_workbook
            workbook = load_workbook(io.BytesIO(content), read_only=True, data_only=True)
            sheet = workbook.active
            values = list(sheet.values)
            headers, records = values[0], values[1:]
        except Exception as exc:
            raise HTTPException(400, f"Invalid XLSX: {exc}")
    else:
        try:
            text = content.decode("utf-8-sig")
            reader = csv.reader(io.StringIO(text)); values = list(reader); headers, records = values[0], values[1:]
        except Exception as exc:
            raise HTTPException(400, f"Invalid CSV: {exc}")
    required = {"activity_code", "description", "planned_start", "planned_finish", "discipline"}
    normalized_headers = {str(h).strip().lower() for h in headers if h is not None}
    missing = required - normalized_headers
    if missing: raise HTTPException(400, f"Missing required columns: {', '.join(sorted(missing))}")
    positions = {str(h).strip().lower(): i for i, h in enumerate(headers)}
    errors, inserted = [], 0
    inserted_codes: dict[str, int] = {}
    pending_dependencies: list[tuple[str, str]] = []
    with closing(connect()) as db:
        for number, row in enumerate(records, 2):
            item = {key: row[index] if index < len(row) else "" for key, index in positions.items()}
            discipline = str(item["discipline"]).strip().lower()
            start, finish = parse_day(item["planned_start"]), parse_day(item["planned_finish"])
            if discipline not in DISCIPLINES: errors.append({"row": number, "error": f"invalid discipline {discipline}"}); continue
            if not start or not finish or finish < start: errors.append({"row": number, "error": "invalid planned dates"}); continue
            code = str(item.get("activity_code", "")).strip()
            if not code: errors.append({"row": number, "error": "activity_code is required"}); continue
            try:
                level = int(item.get("level") or 6)
            except ValueError:
                errors.append({"row": number, "error": "level must be an integer"}); continue
            if level not in {5, 6}: errors.append({"row": number, "error": "only L5/L6 activities are accepted"}); continue
            try:
                cursor = db.execute("INSERT INTO activities(project_id,activity_code,description,wbs_path,level,discipline,location,planned_start,planned_finish,planned_duration,status,contractor,resource) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (project_id, code, item["description"], item.get("wbs_path", ""), level, discipline, item.get("location", ""), start.isoformat(), finish.isoformat(), (finish-start).days+1, "not_started", item.get("contractor", ""), item.get("resource", "")))
                inserted_codes[code] = cursor.lastrowid
                for key in ("parent_activity", "predecessor", "predecessors", "dependency", "dependencies"):
                    for predecessor in re.split(r"[,;|]", str(item.get(key, ""))):
                        if predecessor.strip(): pending_dependencies.append((predecessor.strip(), code))
                inserted += 1
            except sqlite3.IntegrityError as exc: errors.append({"row": number, "error": str(exc)})
        for predecessor, successor in pending_dependencies:
            predecessor_id = inserted_codes.get(predecessor)
            successor_id = inserted_codes.get(successor)
            if not predecessor_id or not successor_id:
                errors.append({"dependency": f"{predecessor} -> {successor}", "error": "unknown dependency activity_code"})
                continue
            db.execute("INSERT INTO activity_dependencies(project_id,predecessor_id,successor_id,dependency_type,lag) VALUES (?,?,?,?,?)", (project_id, predecessor_id, successor_id, "FS", 0))
        db.commit()
        try:
            populate_activity_embeddings(db, project_id)
            recompute(db, project_id, "schedule import")
        except ValueError as exc:
            db.rollback()
            raise HTTPException(400, str(exc))
    return {"inserted": inserted, "errors": errors, "rows": len(records)}
