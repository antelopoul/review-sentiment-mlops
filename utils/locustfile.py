"""
Locust load test for the ONNX text classification API.

Usage:
    pip install locust
    locust -f locustfile.py --host http://localhost:8000

Then open http://localhost:8089 to configure users/rps and start the test.

Headless (CI / scripted) example — 20 users, 5 min, results to CSV:
    locust -f locustfile.py --host http://localhost:8000 \
        --headless -u 20 -r 2 --run-time 5m \
        --csv results/load_test --csv-full-history

Interpreting results:
    p50 < 50 ms   → healthy single-request latency
    p99 < 300 ms  → target SLA
    failure rate  → should stay at 0 %
    RPS           → throughput at your concurrency level
"""

import os
import random
from locust import HttpUser, task, between, events

API_KEY = os.environ.get("API_KEY", "mykey")
HEADERS = {"X-API-Key": API_KEY}

SLA = 300 # p99 latency target in ms
# ---------------------------------------------------------------------------
# Sample inputs — mix of short, medium, and near-limit lengths
# ---------------------------------------------------------------------------

SHORT_TEXTS = [
    "Great product!",
    "Terrible experience.",
    "Works as expected.",
    "Would not recommend.",
    "Five stars!",
]

MEDIUM_TEXTS = [
    "This product is really good and works well. I bought it last month and have been using it daily.",
    "The quality is poor and it broke after two days. Very disappointed with the purchase.",
    "Decent for the price, but the packaging was damaged when it arrived. Customer service was helpful.",
    "Exceeded my expectations! Fast shipping and exactly as described. Will definitely buy again.",
    "Average product. Not bad but not great either. Does the job but nothing special.",
]

LONG_TEXTS = [
    (
        "I have been using this product for over three months now and I have to say it has completely "
        "transformed my daily routine. The build quality is exceptional, the performance is consistent, "
        "and it has never let me down. I was initially skeptical given the price point but after seeing "
        "the results I can confidently say it is worth every penny. Highly recommend to anyone looking "
        "for a reliable and well-made product. The customer support team was also very responsive when "
        "I had a minor setup question in the first week."
    ),
    (
        "Absolutely dreadful. I ordered this based on glowing reviews and was deeply disappointed. "
        "The product arrived damaged, and when I contacted support they were unhelpful and dismissive. "
        "After three weeks of back and forth emails they offered a partial refund that did not even "
        "cover the return shipping cost. The item itself stopped working after just four days of light "
        "use. I would strongly advise anyone reading this to look elsewhere and save yourself the "
        "frustration and wasted money."
    ),
]

# Edge case: exactly at the character limit
BOUNDARY_TEXT = "A" * 1000

# Edge case: one character over the limit — should return 422
OVER_LIMIT_TEXT = "A" * 1001

ALL_VALID_TEXTS = SHORT_TEXTS + MEDIUM_TEXTS + LONG_TEXTS + [BOUNDARY_TEXT]


# ---------------------------------------------------------------------------
# User behaviour
# ---------------------------------------------------------------------------

class PredictUser(HttpUser):
    """
    Simulates a realistic mix of predict requests.
    Wait 0–1 s between tasks to model think-time / downstream processing.
    """

    wait_time = between(0, 1)

    @task(8)
    def predict_valid(self):
        """Happy-path inference — the bulk of traffic."""
        text = random.choice(ALL_VALID_TEXTS)
        with self.client.post(
            "/predict",
            json={"text": text},
            headers=HEADERS,
            catch_response=True,
            name="/predict [valid]",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "prediction" not in data:
                    resp.failure("Response missing 'prediction'")
                else:
                    resp.success()
            else:
                resp.failure(f"Unexpected status {resp.status_code}: {resp.text}")

    @task(1)
    def predict_over_limit(self):
        """Sends text over MAX_INPUT_CHARS — expects 422, not a 5xx."""
        with self.client.post(
            "/predict",
            json={"text": OVER_LIMIT_TEXT},
            headers=HEADERS,
            catch_response=True,
            name="/predict [over limit]",
        ) as resp:
            if resp.status_code == 422:
                resp.success()  # correct rejection
            else:
                resp.failure(f"Expected 422, got {resp.status_code}")

    @task(1)
    def health_check(self):
        """Liveness probe — should always be fast."""
        with self.client.get("/health", catch_response=True, name="/health") as resp:
            if resp.status_code == 200 and resp.json().get("status") == "ok":
                resp.success()
            else:
                resp.failure(f"Health check failed: {resp.status_code}")


# ---------------------------------------------------------------------------
# Custom p99 assertion printed at end of headless run
# ---------------------------------------------------------------------------

@events.quitting.add_listener
def check_sla(environment, **kwargs):
    p99 = environment.stats.total.get_response_time_percentile(0.99)
    target_ms = SLA
    label = "PASS" if p99 is not None and p99 < target_ms else "FAIL"
    print(f"\n[SLA] p99 latency = {p99:.1f} ms  (target < {target_ms} ms)  [{label}]")
    if label == "FAIL":
        environment.process_exit_code = 1
