#!/usr/bin/env python3
"""Benchmark whisper-cli JFK inference and emit bench_output.json."""

import argparse
import json
import math
import os
import platform
import subprocess
import time
from pathlib import Path


def percentile_nearest_rank(sorted_vals: list[float], p: float) -> float:
    n = len(sorted_vals)
    idx = max(0, math.ceil(p / 100.0 * n) - 1)
    return sorted_vals[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run JFK inference benchmark.")
    parser.add_argument("--arm", default="remote-agent")
    parser.add_argument("--branch", default="bench/remote-agent")
    parser.add_argument("--cmake-source", default="system")
    parser.add_argument("--build-wall-s", type=float, required=True)
    parser.add_argument("--ctest-passed", type=int, required=True)
    parser.add_argument("--model-used", default="models/ggml-tiny.en.bin")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", default="bench_output.json")
    args = parser.parse_args()

    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()

    cmd = [
        "./build/bin/whisper-cli",
        "-m",
        args.model_used,
        "-f",
        "samples/jfk.wav",
    ]

    runs_ms: list[float] = []
    verified = False
    for i in range(args.runs):
        t0 = time.perf_counter()
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        runs_ms.append(elapsed_ms)
        if i == 0 or verified:
            verified = "fellow americans" in proc.stdout.lower()

    if args.model_used != "models/ggml-tiny.en.bin":
        verified = False

    sorted_runs = sorted(runs_ms)
    p50_ms = percentile_nearest_rank(sorted_runs, 50)
    p95_ms = percentile_nearest_rank(sorted_runs, 95)

    payload = {
        "arm": args.arm,
        "branch": args.branch,
        "commit_sha": commit_sha,
        "host_os": f"{platform.system()} {platform.machine()}",
        "cpu_count": os.cpu_count(),
        "cmake_source": args.cmake_source,
        "build_wall_s": round(args.build_wall_s, 2),
        "ctest_passed": args.ctest_passed,
        "model_used": args.model_used,
        "jfk_transcript_verified": verified,
        "runs_ms": runs_ms,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
    }

    Path(args.output).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
