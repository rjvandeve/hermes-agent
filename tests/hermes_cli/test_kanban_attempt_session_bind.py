"""Real-path attempt session writer → R2 consumer contract.

Binds worker_session_id onto task_runs.metadata early (not completion-only).
Drives a minimal R2-shaped reader against writer-produced rows — no pre-seeded
imaginary links. Public fixtures only (no private paths/credentials).
"""
from __future__ import annotations

import json
import os

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from tools import kanban_tools as kt


@pytest.fixture
def board(tmp_path, monkeypatch):
    db = tmp_path / "kanban.db"
    # kanban_home ignores HERMES_HOME — pin DB so bind path hits the fixture.
    monkeypatch.setenv("HERMES_KANBAN_DB", str(db))
    monkeypatch.setenv("HERMES_KANBAN_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_RUN_ID", raising=False)
    monkeypatch.delenv("HERMES_SESSION_ID", raising=False)
    monkeypatch.delenv("HERMES_PROFILE", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    conn = kbc.connect(db)
    try:
        yield conn
    finally:
        conn.close()


def _claim_run(conn, *, title: str, assignee: str = "momentum-worker") -> tuple[str, int]:
    tid = kb.create_task(conn, title=title, assignee=assignee)
    kb.claim_task(conn, tid)
    row = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row and row["current_run_id"]
    return tid, int(row["current_run_id"])


def _meta(conn, run_id: int) -> dict:
    row = conn.execute("SELECT metadata FROM task_runs WHERE id = ?", (run_id,)).fetchone()
    if not row or not row["metadata"]:
        return {}
    return json.loads(row["metadata"])


def _r2_link_from_runs(conn) -> dict:
    """Minimal R2 consumer: map worker_session_id → (task_id, run_id)."""
    maps = {}
    for task_id, run_id, meta in conn.execute(
        "SELECT task_id, id, metadata FROM task_runs ORDER BY id"
    ):
        wsid = None
        if meta:
            try:
                wsid = json.loads(meta).get("worker_session_id")
            except Exception:
                wsid = None
        if wsid:
            maps[str(wsid)] = {"task_id": task_id, "run_id": int(run_id), "link_method": "run.metadata.worker_session_id"}
    return maps


def test_two_attempts_first_fails_second_succeeds(board, monkeypatch):
    conn = board
    tid, run1 = _claim_run(conn, title="job")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run1))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_attempt1_fail")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-worker")
    r = kt.bind_attempt_session_from_env("sess_attempt1_fail")
    assert r["ok"] is True
    assert _meta(conn, run1)["worker_session_id"] == "sess_attempt1_fail"

    # Crash ends run1 — session must survive on the closed row.
    # Mirror detect_crashed_workers: end run + re-ready for retry.
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed",
                    error="boom", metadata={"pid": 1, "exit_kind": "signal"})
        conn.execute(
            "UPDATE tasks SET status = 'ready', claim_lock = NULL, "
            "claim_expires = NULL, worker_pid = NULL, worker_started_at = NULL "
            "WHERE id = ?",
            (tid,),
        )
    m1 = _meta(conn, run1)
    assert m1["worker_session_id"] == "sess_attempt1_fail"
    assert m1.get("exit_kind") == "signal"

    # Second attempt
    claimed = kb.claim_task(conn, tid)
    assert claimed is not None
    run2 = int(conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (tid,)).fetchone()[0])
    assert run2 != run1
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run2))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_attempt2_ok")
    r2 = kt.bind_attempt_session_from_env("sess_attempt2_ok")
    assert r2["ok"] is True
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="completed", status="done",
                    summary="done", metadata={"note": "ok"})
    m2 = _meta(conn, run2)
    assert m2["worker_session_id"] == "sess_attempt2_ok"

    links = _r2_link_from_runs(conn)
    assert links["sess_attempt1_fail"]["run_id"] == run1
    assert links["sess_attempt2_ok"]["run_id"] == run2
    assert links["sess_attempt1_fail"]["task_id"] == tid


def test_crash_after_session_init_keeps_link(board, monkeypatch):
    conn = board
    tid, run_id = _claim_run(conn, title="crash-after-init")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_crash")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-worker")
    assert kt.bind_attempt_session_from_env()["ok"] is True
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed",
                    error="segfault", metadata={"exit_code": -11})
    assert _meta(conn, run_id)["worker_session_id"] == "sess_crash"
    links = _r2_link_from_runs(conn)
    assert links["sess_crash"]["link_method"] == "run.metadata.worker_session_id"


def test_reviewer_vs_implementer_identity(board, monkeypatch):
    conn = board
    tid_i, run_i = _claim_run(conn, title="impl", assignee="momentum-worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid_i)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_i))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_impl")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-worker")
    assert kt.bind_attempt_session_from_env()["ok"] is True
    assert _meta(conn, run_i).get("worker_profile") == "momentum-worker"

    tid_r, run_r = _claim_run(conn, title="review", assignee="momentum-reviewer")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid_r)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_r))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_rev")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-reviewer")
    assert kt.bind_attempt_session_from_env()["ok"] is True
    assert _meta(conn, run_r)["worker_session_id"] == "sess_rev"
    assert _meta(conn, run_r).get("worker_profile") == "momentum-reviewer"
    # implementer row untouched
    assert _meta(conn, run_i)["worker_session_id"] == "sess_impl"


def test_stale_run_refusal(board, monkeypatch):
    conn = board
    tid, run_id = _claim_run(conn, title="stale")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_ok")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-worker")
    assert kt.bind_attempt_session_from_env()["ok"] is True

    # End the run, then refuse bind with the stale run id still in env.
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed", error="x")
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_late")
    r = kt.bind_attempt_session_from_env("sess_late")
    assert r["ok"] is False
    assert r["reason"] == "stale_or_foreign_run"
    # original session preserved; late session not written
    assert _meta(conn, run_id)["worker_session_id"] == "sess_ok"


def test_foreign_profile_refusal(board, monkeypatch):
    conn = board
    tid, run_id = _claim_run(conn, title="foreign", assignee="momentum-worker")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_x")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-reviewer")  # mismatch assignee
    r = kt.bind_attempt_session_from_env()
    assert r["ok"] is False
    assert r["reason"] == "stale_or_foreign_run"
    assert "worker_session_id" not in _meta(conn, run_id)


def test_no_session_yet_honest(board, monkeypatch):
    conn = board
    tid, run_id = _claim_run(conn, title="pre-session")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    # no HERMES_SESSION_ID
    r = kt.bind_attempt_session_from_env(None)
    assert r["ok"] is False
    assert r["reason"] == "no_session_yet"
    assert _meta(conn, run_id) == {}


def test_mutation_without_merge_would_wipe_on_crash(board, monkeypatch):
    """Discriminating: if _end_run replaced metadata instead of merging, session dies."""
    conn = board
    tid, run_id = _claim_run(conn, title="mut")
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_SESSION_ID", "sess_keep")
    monkeypatch.setenv("HERMES_PROFILE", "momentum-worker")
    assert kt.bind_attempt_session_from_env()["ok"] is True
    with kb.write_txn(conn):
        kb._end_run(conn, tid, outcome="crashed", status="crashed",
                    error="x", metadata={"only": "exit"})
    # If merge broken → worker_session_id gone. This asserts merge works.
    assert _meta(conn, run_id).get("worker_session_id") == "sess_keep"
    assert _meta(conn, run_id).get("only") == "exit"
