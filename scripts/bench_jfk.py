#!/usr/bin/env python3
"""Sequential JFK inference benchmark runner for whisper.cpp."""

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path


def percentile(sorted_vals: list[float], p: int) -> float:
    """Return nearest-rank percentile for sorted_vals."""
    n = len(sorted_vals)
    idx = max(0, math.ceil(p / 100 * n) - 1)
    return sorted_vals[idx]


def git_output(args: list[str]) -> str:
    """Return stripped stdout from a git command."""
    res = subprocess.run(["git", *args], check=True, capture_output=True, text=True)
    return res.stdout.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark whisper-cli on samples/jfk.wav")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--cmake-source", required=True)
    parser.add_argument("--build-wall-s", type=float, required=True)
    parser.add_argument("--ctest-passed", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="bench_output.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    cmd = ["./build/bin/whisper-cli", "-m", args.model, "-f", "samples/jfk.wav"]
    runs_ms: list[float] = []
    verified_all = True

    for i in range(args.runs):
        t0 = time.perf_counter()
        proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper-cli run {i + 1} failed with code {proc.returncode}: {proc.stderr}"
            )
        combined = f"{proc.stdout}\n{proc.stderr}".lower()
        if "fellow americans" not in combined:
            verified_all = False
        runs_ms.append(elapsed_ms)

    sorted_runs = sorted(runs_ms)
    payload = {
        "arm": args.arm,
        "branch": git_output(["branch", "--show-current"]),
        "commit_sha": git_output(["rev-parse", "HEAD"]),
        "host_os": f"{platform.system()} {platform.machine()}",
        "cpu_count": os.cpu_count(),
        "cmake_source": args.cmake_source,
        "build_wall_s": round(args.build_wall_s, 2),
        "ctest_passed": args.ctest_passed,
        "model_used": args.model,
        "jfk_transcript_verified": verified_all,
        "runs_ms": runs_ms,
        "p50_ms": percentile(sorted_runs, 50),
        "p95_ms": percentile(sorted_runs, 95),
    }

    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
