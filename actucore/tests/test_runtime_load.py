"""The synthetic placement load tool, run the way CI runs it.

The tool must stand alone: no GPU, no external network, no pytest fixtures
inside the process it starts. Running it twice also proves the decision table
is repeatable rather than timing-dependent.

Run: cd actucore && PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest \
     tests/test_runtime_load.py -q
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "runtime_load.py"


def run_tool(*args):
    return subprocess.run(
        [sys.executable, str(TOOL), "--json", *args],
        cwd=str(ROOT), capture_output=True, text=True, timeout=180)


def test_runtime_load_cli_reports_ten_of_ten_expected_decisions():
    result = run_tool()
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["scenario"] == "local-first-placement"
    assert report["summary"]["total"] == 10
    assert report["summary"]["passed"] == 10
    assert report["summary"]["failed"] == 0
    assert report["summary"]["ok"] is True
    assert len(report["decisions"]) == 10
    assert all(decision["ok"] for decision in report["decisions"])
    assert {decision["target"] for decision in report["decisions"]} == \
        {"local", "remote"}


def test_runtime_load_is_repeatable():
    targets = []
    for _ in range(2):
        result = run_tool()
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["summary"]["ok"] is True
        targets.append([decision["target"] for decision in report["decisions"]])
    assert targets[0] == targets[1]


def test_runtime_load_cli_reports_five_of_five_failovers():
    result = run_tool("--scenario", "failover")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)

    assert report["scenario"] == "mid-session-failover"
    assert report["summary"]["total"] == 5
    assert report["summary"]["passed"] == 5
    assert report["summary"]["failed"] == 0
    assert report["summary"]["ok"] is True
    assert len(report["failovers"]) == 5
    assert all(failover["ok"] for failover in report["failovers"])
    assert all(failover["target"] == "remote"
               for failover in report["failovers"])
    assert all(failover["failover_count"] == 1
               for failover in report["failovers"])
    assert report["counters"]["failover_count"] == 5
    assert report["counters"]["failback_count"] == 1
    assert report["counters"]["switch_count"] == 6
    assert report["failback"]["target"] == "local"
    assert report["failback"]["ok"] is True
    assert report["sticky"]["ok"] is True


def test_runtime_load_failover_is_repeatable():
    targets = []
    for _ in range(2):
        result = run_tool("--scenario", "failover")
        assert result.returncode == 0, result.stderr
        report = json.loads(result.stdout)
        assert report["summary"]["ok"] is True
        targets.append([failover["target"] for failover in report["failovers"]])
    assert targets[0] == targets[1]
