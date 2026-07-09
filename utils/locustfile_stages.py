"""
Staged-load Locust file for throughput/latency-vs-load plots.

Ramps through fixed load levels (STAGES below) and holds each one long
enough to reach steady state. Each simulated user paces itself to ~1
request/second (constant_pacing), so "load" == user count == target RPS —
until the server can no longer keep up, at which point achieved throughput
falls behind the target and latency climbs. That divergence is the point:
scripts/plot_locust_results.py plots it as throughput-vs-load,
latency-vs-load, and throughput-vs-latency.

Usage:
    pip install locust
    locust -f utils/locustfile_stages.py --host http://localhost:8000 --headless

    (the shape below drives users/duration — -u/-r/--run-time are ignored)

Then:
    python scripts/plot_locust_results.py

Outputs:
    results/locust_raw_requests.csv — one row per request (gitignored)
"""

import csv
import os
import time

from locust import HttpUser, LoadTestShape, constant_pacing, events, task

from utils.locustfile import ALL_VALID_TEXTS, HEADERS
import random

RAW_OUTPUT = os.environ.get("LOCUST_RAW_OUTPUT", "results/locust_raw_requests.csv")

# (users, duration_seconds, spawn_rate) — duration is how long each stage
# is held, not cumulative. Kept in sync with the color-by-load plot.
STAGES = [
    {"users": 10,  "duration": 30, "spawn_rate": 10},
    {"users": 50,  "duration": 45, "spawn_rate": 20},
    {"users": 100, "duration": 45, "spawn_rate": 20},
    {"users": 200, "duration": 45, "spawn_rate": 20},
    {"users": 300, "duration": 45, "spawn_rate": 20},
    {"users": 400, "duration": 45, "spawn_rate": 20},
]

# Seconds of ramp-up/settling to discard from the front of each stage
# before treating requests as steady-state.
WARMUP_S = 20


class PacedPredictUser(HttpUser):
    """One /predict call per second per user, until the server can't keep up."""

    wait_time = constant_pacing(1)

    @task
    def predict_valid(self):
        text = random.choice(ALL_VALID_TEXTS)
        with self.client.post(
            "/predict",
            json={"text": text},
            headers=HEADERS,
            catch_response=True,
            name="/predict",
        ) as resp:
            if resp.status_code == 200 and "prediction" in resp.json():
                resp.success()
            else:
                resp.failure(f"status={resp.status_code}")


class StagedRampShape(LoadTestShape):
    """Steps through STAGES, holding each user count for its duration."""

    def tick(self):
        run_time = self.get_run_time()
        elapsed = 0
        for stage in STAGES:
            elapsed += stage["duration"]
            if run_time < elapsed:
                return stage["users"], stage["spawn_rate"]
        return None


# ---------------------------------------------------------------------------
# Raw per-request logging — the stats-history CSV Locust writes with
# --csv-full-history reports *cumulative* percentiles, which hides
# per-stage latency spikes. Logging every request lets the plot script
# compute true per-stage throughput/percentiles instead.
# ---------------------------------------------------------------------------

_test_start = 0.0
_rows: list[tuple[float, float, bool]] = []


@events.test_start.add_listener
def _on_test_start(environment, **kwargs):
    global _test_start
    _test_start = time.time()
    _rows.clear()


@events.request.add_listener
def _on_request(request_type, name, response_time, response_length, response,
                 context, exception, start_time=None, **kwargs):
    elapsed = (start_time if start_time else time.time()) - _test_start
    _rows.append((elapsed, response_time, exception is None))


@events.test_stop.add_listener
def _on_test_stop(environment, **kwargs):
    os.makedirs(os.path.dirname(RAW_OUTPUT), exist_ok=True)
    with open(RAW_OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["elapsed_s", "response_time_ms", "success"])
        writer.writerows(_rows)
    print(f"\nRaw request log saved -> {RAW_OUTPUT} ({len(_rows)} requests)")
