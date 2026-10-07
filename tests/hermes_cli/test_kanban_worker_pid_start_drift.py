"""Kanban composite fingerprint liveness tolerates macOS ±100cs start drift (#117505).

Missed site vs cron/delivery_ledger/api_server_runs/async_delegation: exact string
equality on ``epoch|start`` false-deaded live workers. Uses owned temporary process
identity only — never signals a real worker/gateway.
"""
from __future__ import annotations

import os

import pytest

from gateway.status import START_TIME_DRIFT_TOLERANCE, start_time_fingerprints_match
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def live_fp(monkeypatch):
    """Stable composite fingerprint for this process; controllable probe."""
    epoch = "boot-test"
    start = 1_791_341_454_55  # centiseconds-like
    live = f"{epoch}|{start}"

    def _fp(pid):
        if int(pid) != os.getpid():
            return None
        return getattr(_fp, "current", live)

    _fp.current = live
    monkeypatch.setattr(kbd, "_process_fingerprint", _fp)
    monkeypatch.setattr(kbd._kb, "_pid_alive", lambda pid: int(pid) == os.getpid())
    return {"epoch": epoch, "start": start, "live": live, "set": lambda v: setattr(_fp, "current", v)}


def test_exact_match_alive(live_fp):
    assert kbd._worker_alive(os.getpid(), live_fp["live"]) is True
    assert kbd._pid_recycled(os.getpid(), live_fp["live"]) is False


@pytest.mark.parametrize("delta", [100, -100])
def test_plus_minus_100_centis_alive(live_fp, delta):
    recorded = f"{live_fp['epoch']}|{live_fp['start']}"
    live_fp["set"](f"{live_fp['epoch']}|{live_fp['start'] + delta}")
    assert start_time_fingerprints_match(live_fp["start"], live_fp["start"] + delta)
    assert kbd._worker_alive(os.getpid(), recorded) is True
    assert kbd._pid_recycled(os.getpid(), recorded) is False


def test_beyond_tolerance_dead(live_fp):
    recorded = f"{live_fp['epoch']}|{live_fp['start']}"
    far = live_fp["start"] + START_TIME_DRIFT_TOLERANCE + 1
    live_fp["set"](f"{live_fp['epoch']}|{far}")
    assert not start_time_fingerprints_match(live_fp["start"], far)
    assert kbd._worker_alive(os.getpid(), recorded) is False
    assert kbd._pid_recycled(os.getpid(), recorded) is True


def test_foreign_epoch_dead(live_fp):
    recorded = f"other-boot|{live_fp['start']}"
    assert kbd._worker_alive(os.getpid(), recorded) is False
    assert kbd._pid_recycled(os.getpid(), recorded) is True


def test_malformed_fingerprint_fail_closed(live_fp):
    assert kbd._pid_recycled(os.getpid(), "not-a-fingerprint|") is True
    assert kbd._pid_recycled(os.getpid(), f"{live_fp['epoch']}|") is True
    # unavailable current fingerprint
    live_fp["set"](None)
    assert kbd._pid_recycled(os.getpid(), live_fp["live"]) is True


def test_recycled_start_far_dead(live_fp):
    """Different process incarnation: start far outside tolerance."""
    recorded = f"{live_fp['epoch']}|{live_fp['start']}"
    live_fp["set"](f"{live_fp['epoch']}|{live_fp['start'] + 3600}")
    assert kbd._worker_alive(os.getpid(), recorded) is False


def test_unverified_still_existence_only_for_alive_path(live_fp):
    # UNVERIFIED: _worker_alive True if pid alive; _pid_recycled True (never signal).
    assert kbd._worker_alive(os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT) is True
    assert kbd._pid_recycled(os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT) is True


def test_mutation_exact_equality_would_false_dead(live_fp, monkeypatch):
    """Discriminating mutation: restore exact-string compare → ±100 fails."""
    recorded = f"{live_fp['epoch']}|{live_fp['start']}"
    live_fp["set"](f"{live_fp['epoch']}|{live_fp['start'] + 100}")

    def exact(pid, started_at):
        if started_at is None or not pid:
            return False
        if started_at == kbd.UNVERIFIED_WORKER_FINGERPRINT:
            return True
        if isinstance(started_at, str) and "|" in started_at:
            return kbd._process_fingerprint(int(pid)) != started_at
        return True

    monkeypatch.setattr(kbd, "_pid_recycled", exact)
    # With exact equality the drifted worker is false-dead — proves the test catches regress.
    assert kbd._worker_alive(os.getpid(), recorded) is False
