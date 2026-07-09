"""
Plot throughput/latency-vs-load charts from a staged Locust run.

Generates three panels matching the Trustpilot reference:
  1. Throughput vs Load (RPS)
  2. P99 Latency vs Load (RPS)
  3. Throughput vs P99 Latency (scatter colored by P99)

Usage:
    locust -f utils/locustfile_stages.py --host http://localhost:8000 --headless
    python scripts/plot_locust_results.py

Reads:
    results/locust_raw_requests.csv — per-request log written by
    utils/locustfile_stages.py (elapsed_s, response_time_ms, success)

Outputs:
    results/locust_stage_<date>.json — per-stage summary (committed)
    results/locust_stage_<date>.png  — 3-panel chart (gitignored)
"""

import csv
import json
import os
import sys
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from utils.locustfile_stages import STAGES, WARMUP_S

RAW_INPUT = os.environ.get("LOCUST_RAW_OUTPUT", "results/locust_raw_requests.csv")

# Sequential blue ramp, light -> dark (see dataviz skill: references/palette.md)
BLUE_SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue_sequential",
    ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"],
)


# ---------------------------------------------------------------------------
# Load raw requests and bucket into per-stage steady-state windows
# ---------------------------------------------------------------------------

def load_requests(path: str) -> list[tuple[float, float, bool]]:
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for elapsed_s, response_time_ms, success in reader:
            rows.append((float(elapsed_s), float(response_time_ms), success == "True"))
    return rows


def stage_windows() -> list[tuple[int, float, float]]:
    """Returns (target_load, window_start_s, window_end_s) per stage,
    excluding the WARMUP_S ramp-up/settling period at the front of each."""
    windows = []
    t = 0.0
    for stage in STAGES:
        end = t + stage["duration"]
        windows.append((stage["users"], t + WARMUP_S, end))
        t = end
    return windows


def summarize(rows: list[tuple[float, float, bool]]) -> list[dict]:
    results = []
    for load, start, end in stage_windows():
        window = [r for r in rows if start <= r[0] < end]
        latencies = np.array([r[1] for r in window]) if window else np.array([])
        n = len(window)
        duration = end - start
        successes = sum(1 for r in window if r[2])
        results.append({
            "load": load,
            "n_requests": n,
            "throughput_rps": round(n / duration, 2) if duration > 0 else 0.0,
            "p50_ms": round(float(np.percentile(latencies, 50)), 1) if n else 0.0,
            "p95_ms": round(float(np.percentile(latencies, 95)), 1) if n else 0.0,
            "p99_ms": round(float(np.percentile(latencies, 99)), 1) if n else 0.0,
            "failure_pct": round(100 * (1 - successes / n), 1) if n else 0.0,
        })
    return results


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot(results: list[dict], output: str):
    load = [r["load"] for r in results]
    throughput = [r["throughput_rps"] for r in results]
    p99_s = [r["p99_ms"] / 1000 for r in results]

    SLA_MS = 300  # SLA target in milliseconds
    SLA_S = SLA_MS / 1000

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Locust staged load test", fontsize=14, fontweight="bold")

    # --- Panel 1: Throughput vs Load ---
    ideal_max = max(load)
    ax1.plot([0, ideal_max], [0, ideal_max], linestyle="--", color="#c3c2b7",
              linewidth=1.5, label="Ideal (100%)")
    ax1.plot(load, throughput, marker="o", linewidth=2, color="#2a78d6")
    ax1.set_title("Throughput vs Load")
    ax1.set_xlabel("Load (requests/second)")
    ax1.set_ylabel("Throughput (requests/second)")
    ax1.grid(axis="both", color="#e1e0d9", linewidth=0.8)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # --- Panel 2: Latency vs Load ---
    ax2.plot(load, p99_s, marker="o", linewidth=2, color="#e34948")
    ax2.axhline(SLA_S, color="#eda100", linewidth=1.5, linestyle=":", label=f"{SLA_MS}ms SLA")
    ax2.fill_between(load, 0, SLA_S, alpha=0.04, color="#1baf7a")
    ax2.set_title("Latency (P99) vs Load")
    ax2.set_xlabel("Load (requests/second)")
    ax2.set_ylabel("P99 Latency (seconds)")
    ax2.grid(axis="both", color="#e1e0d9", linewidth=0.8)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # --- Panel 3: Throughput vs Latency, colored by P99 latency ---
    sizes = [40 + 260 * (l / ideal_max) for l in load]
    sc = ax3.scatter(p99_s, throughput, c=p99_s, s=sizes, cmap=BLUE_SEQUENTIAL,
                      edgecolors="#0b0b0b", linewidths=0.6, zorder=3)
    for x, y, l in zip(p99_s, throughput, load):
        ax3.annotate(f"{l} rps", (x, y), textcoords="offset points",
                     xytext=(8, 4), fontsize=8, color="#52514e")
    ax3.set_title("Throughput vs Latency (P99)")
    ax3.set_xlabel("P99 Latency (seconds)")
    ax3.set_ylabel("Throughput (requests/second)")
    ax3.grid(axis="both", color="#e1e0d9", linewidth=0.8)
    ax3.spines[["top", "right"]].set_visible(False)
    cbar = fig.colorbar(sc, ax=ax3)
    cbar.set_label("P99 Latency (s)")

    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"\nChart saved -> {output}")
    plt.show()


def save_json(results: list[dict], output: str):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    data = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "warmup_s": WARMUP_S,
        "stages": results,
    }
    with open(output, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved -> {output}")


def main():
    if not os.path.exists(RAW_INPUT):
        print(f"ERROR: {RAW_INPUT} not found — run the staged Locust test first:\n"
              f"  locust -f utils/locustfile_stages.py --host http://localhost:8000 --headless")
        sys.exit(1)

    rows = load_requests(RAW_INPUT)
    results = summarize(rows)

    print(f"{'Load':>6}  {'Requests':>8}  {'Throughput':>10}  {'p99':>8}  {'Fail%':>6}")
    print("-" * 58)
    for r in results:
        print(f"{r['load']:>6}  {r['n_requests']:>8}  {r['throughput_rps']:>9.1f}r/s  "
              f"{r['p99_ms']:>7.1f}ms  {r['failure_pct']:>5.1f}%")

    date_tag = datetime.now().strftime("%Y-%m-%d")
    save_json(results, f"results/locust_stage_{date_tag}.json")
    plot(results, f"results/locust_stage_{date_tag}.png")


if __name__ == "__main__":
    main()
