#!/usr/bin/env python3
"""
Report shared NVIDIA GPU load and select a device for a CI job.
"""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

GPU_SAMPLE_QUERY = "index,utilization.gpu,memory.used,memory.total"
GPU_DETAIL_QUERY = (
    "timestamp,index,uuid,name,pstate,utilization.gpu,utilization.memory,"
    "memory.used,memory.total,power.draw,power.limit,temperature.gpu,"
    "temperature.memory,clocks.current.sm,clocks.max.sm,"
    "clocks_event_reasons.active,compute_mode"
)
COMPUTE_APP_QUERY = "timestamp,gpu_uuid,pid,process_name,used_gpu_memory"


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuSample:
    index: int
    utilization: float
    memory_used: int
    memory_total: int


@dataclass
class GpuSummary:
    index: int
    utilization_sum: float = 0.0
    peak_utilization: float = 0.0
    max_memory_used: int = 0
    min_memory_free: int | None = None
    sample_count: int = 0

    @property
    def average_utilization(self) -> float:
        return self.utilization_sum / self.sample_count

    def add(self, sample: GpuSample) -> None:
        self.utilization_sum += sample.utilization
        self.peak_utilization = max(self.peak_utilization, sample.utilization)
        self.max_memory_used = max(self.max_memory_used, sample.memory_used)
        memory_free = sample.memory_total - sample.memory_used
        if self.min_memory_free is None:
            self.min_memory_free = memory_free
        else:
            self.min_memory_free = min(self.min_memory_free, memory_free)
        self.sample_count += 1


def print_command(command: list[str], suffix: str = "") -> None:
    print(f"+ {shlex.join(command)}{suffix}", flush=True)


def run_diagnostic(command: list[str]) -> subprocess.CompletedProcess[str]:
    print_command(command)
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if result.stdout:
        print(result.stdout.rstrip())
    if result.returncode:
        print(
            f"WARNING: diagnostic command exited with status {result.returncode}",
            file=sys.stderr,
        )
    return result


def parse_samples(output: str) -> list[GpuSample]:
    samples = []
    for row in csv.reader(output.splitlines()):
        fields = [field.strip() for field in row]
        if not fields:
            continue
        if len(fields) != 4:
            raise PreflightError(f"Unexpected nvidia-smi sample row: {row!r}")
        try:
            samples.append(
                GpuSample(
                    index=int(fields[0]),
                    utilization=float(fields[1]),
                    memory_used=int(fields[2]),
                    memory_total=int(fields[3]),
                )
            )
        except ValueError as error:
            raise PreflightError(
                f"Non-numeric value in nvidia-smi sample row: {row!r}"
            ) from error
    if not samples:
        raise PreflightError("No NVIDIA GPU was discovered")
    return samples


def collect_samples(count: int, interval: float) -> list[GpuSample]:
    command = [
        "nvidia-smi",
        f"--query-gpu={GPU_SAMPLE_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    print_command(command, f"  # {count} samples, {interval:g} seconds apart")
    samples = []
    for sample_number in range(count):
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            raise PreflightError(f"nvidia-smi sampling failed: {detail}")
        samples.extend(parse_samples(result.stdout))
        if sample_number + 1 < count:
            time.sleep(interval)
    return samples


def summarize_samples(samples: list[GpuSample]) -> list[GpuSummary]:
    summaries: dict[int, GpuSummary] = {}
    for sample in samples:
        summary = summaries.setdefault(sample.index, GpuSummary(index=sample.index))
        summary.add(sample)
    return [summaries[index] for index in sorted(summaries)]


def print_samples(samples: list[GpuSample], summaries: list[GpuSummary]) -> None:
    print("GPU selection samples (index, utilization %, used MiB, total MiB):")
    for sample in samples:
        print(
            f"{sample.index}, {sample.utilization:g}, "
            f"{sample.memory_used}, {sample.memory_total}"
        )

    print(
        "GPU selection summary (index, average util %, peak util %, "
        "max used MiB, min free MiB):"
    )
    for summary in summaries:
        print(
            f"{summary.index} {summary.average_utilization:.2f} "
            f"{summary.peak_utilization:g} {summary.max_memory_used} "
            f"{summary.min_memory_free}"
        )


def select_gpu(
    summaries: list[GpuSummary],
    requested_gpu: str,
    min_free_memory: int,
) -> GpuSummary:
    if requested_gpu:
        try:
            requested_index = int(requested_gpu)
        except ValueError as error:
            raise PreflightError(
                f"Requested GPU_DEVICE={requested_gpu!r} is not a physical GPU index"
            ) from error
        for summary in summaries:
            if summary.index == requested_index:
                print(f"Honoring requested physical GPU_DEVICE={requested_index}")
                return summary
        raise PreflightError(
            f"Requested GPU_DEVICE={requested_index} does not identify an available GPU"
        )

    eligible = [
        summary
        for summary in summaries
        if summary.min_memory_free is not None
        and summary.min_memory_free >= min_free_memory
    ]
    if not eligible:
        print(
            f"WARNING: no GPU has {min_free_memory} MiB free; "
            "selecting the least-loaded device",
            file=sys.stderr,
        )
        eligible = summaries

    return min(
        eligible,
        key=lambda summary: (
            summary.average_utilization,
            summary.peak_utilization,
            summary.max_memory_used,
            summary.index,
        ),
    )


def report_process_owners() -> None:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid",
        "--format=csv,noheader,nounits",
    ]
    print_command(command)
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        print(result.stderr.rstrip(), file=sys.stderr)
        return

    pids = sorted(
        {
            int(line.strip())
            for line in result.stdout.splitlines()
            if line.strip().isdigit()
        }
    )
    if pids:
        run_diagnostic(
            [
                "ps",
                "-o",
                "user:24,pid,etimes,comm",
                "-p",
                ",".join(str(pid) for pid in pids),
            ]
        )


def run_monitoring(sample_count: int) -> None:
    commands = [
        [
            "nvidia-smi",
            "dmon",
            "-s",
            "pucvmet",
            "-d",
            "1",
            "-c",
            str(sample_count),
            "-o",
            "DT",
        ],
        [
            "nvidia-smi",
            "pmon",
            "-s",
            "um",
            "-d",
            "1",
            "-c",
            str(sample_count),
            "-o",
            "DT",
        ],
    ]
    processes = []
    for command in commands:
        print_command(command)
        processes.append(
            (
                command,
                subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                ),
            )
        )
    for command, process in processes:
        output, _ = process.communicate()
        print(f"=== {shlex.join(command)} ===")
        if output:
            print(output.rstrip())
        if process.returncode:
            print(
                f"WARNING: diagnostic command exited with status {process.returncode}",
                file=sys.stderr,
            )


def write_environment(path: Path, values: dict[str, str]) -> None:
    content = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in values.items()
    )
    path.write_text(content, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report NVIDIA GPU load and select a device for a CI job."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="write shell exports for the calling CI job to this file",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--monitor-samples", type=int, default=5)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--busy-threshold", type=float, default=80.0)
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_arguments()
    if args.samples < 1 or args.interval < 0 or args.monitor_samples < 1:
        raise PreflightError("Sample counts must be positive and interval non-negative")

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    requested_gpu = os.environ.get("GPU_DEVICE", "")
    print("=== Shared GPU CI preflight ===")
    print(datetime.now().astimezone().isoformat(timespec="seconds"))
    print(socket.getfqdn())
    print(
        f"Initial CUDA_VISIBLE_DEVICES={visible_devices or '<unset>'} "
        f"GPU_DEVICE={requested_gpu or '<unset>'}"
    )

    run_diagnostic(["nvidia-smi"])
    run_diagnostic(["nvidia-smi", "topo", "-m"])
    run_diagnostic(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_DETAIL_QUERY}",
            "--format=csv",
        ]
    )
    run_diagnostic(
        [
            "nvidia-smi",
            f"--query-compute-apps={COMPUTE_APP_QUERY}",
            "--format=csv",
        ]
    )
    report_process_owners()

    if visible_devices:
        print(f"Keeping preconfigured CUDA_VISIBLE_DEVICES={visible_devices}")
        environment = {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "GPU_DEVICE": requested_gpu or "0",
        }
    else:
        samples = collect_samples(args.samples, args.interval)
        summaries = summarize_samples(samples)
        print_samples(samples, summaries)
        selected = select_gpu(summaries, requested_gpu, args.min_free_mib)
        if selected.average_utilization >= args.busy_threshold:
            print(
                f"WARNING: selected GPU {selected.index} averaged "
                f"{selected.average_utilization:.2f}% utilization",
                file=sys.stderr,
            )
        environment = {
            "DARTS_CI_PHYSICAL_GPU_DEVICE": str(selected.index),
            "CUDA_VISIBLE_DEVICES": str(selected.index),
            "GPU_DEVICE": "0",
        }

    run_monitoring(args.monitor_samples)
    write_environment(args.env_file, environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
