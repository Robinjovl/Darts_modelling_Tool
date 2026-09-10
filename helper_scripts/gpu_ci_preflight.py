#!/usr/bin/env python3
"""
Summarize shared NVIDIA GPU occupancy and select a device for a CI job.
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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

GPU_SAMPLE_QUERY = "index,uuid,utilization.gpu,memory.used,memory.total"
COMPUTE_APP_QUERY = "gpu_uuid,pid"


class PreflightError(RuntimeError):
    pass


@dataclass(frozen=True)
class GpuSample:
    index: int
    uuid: str
    utilization: float
    memory_used: int
    memory_total: int


@dataclass
class GpuSummary:
    index: int
    uuid: str
    utilization_sum: float = 0.0
    peak_utilization: float = 0.0
    max_memory_used: int = 0
    memory_total: int = 0
    min_memory_free: int | None = None
    sample_count: int = 0

    @property
    def average_utilization(self) -> float:
        return self.utilization_sum / self.sample_count

    def add(self, sample: GpuSample) -> None:
        self.utilization_sum += sample.utilization
        self.peak_utilization = max(self.peak_utilization, sample.utilization)
        self.max_memory_used = max(self.max_memory_used, sample.memory_used)
        self.memory_total = sample.memory_total
        memory_free = sample.memory_total - sample.memory_used
        if self.min_memory_free is None:
            self.min_memory_free = memory_free
        else:
            self.min_memory_free = min(self.min_memory_free, memory_free)
        self.sample_count += 1


@dataclass
class ProcessSummary:
    count: int = 0
    owners: set[str] = field(default_factory=set)


def parse_samples(output: str) -> list[GpuSample]:
    samples = []
    for row in csv.reader(output.splitlines()):
        fields = [value.strip() for value in row]
        if not fields:
            continue
        if len(fields) != 5:
            raise PreflightError(f"Unexpected nvidia-smi sample row: {row!r}")
        try:
            samples.append(
                GpuSample(
                    index=int(fields[0]),
                    uuid=fields[1],
                    utilization=float(fields[2]),
                    memory_used=int(fields[3]),
                    memory_total=int(fields[4]),
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
    print(f"Sampling {count} times at {interval:g}-second intervals...")
    samples = []
    for sample_number in range(count):
        result = subprocess.run(command, check=False, capture_output=True, text=True)
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
        summary = summaries.setdefault(
            sample.index, GpuSummary(index=sample.index, uuid=sample.uuid)
        )
        summary.add(sample)
    return [summaries[index] for index in sorted(summaries)]


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


def query_process_owners(pids: set[int]) -> dict[int, str]:
    if not pids:
        return {}
    command = [
        "ps",
        "-o",
        "pid=,user=",
        "-p",
        ",".join(str(pid) for pid in sorted(pids)),
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    owners = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit():
            owners[int(fields[0])] = fields[1]
    return owners


def query_processes(summaries: list[GpuSummary]) -> dict[int, ProcessSummary]:
    command = [
        "nvidia-smi",
        f"--query-compute-apps={COMPUTE_APP_QUERY}",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        print("WARNING: could not query active CUDA processes")
        return {}

    processes = []
    for row in csv.reader(result.stdout.splitlines()):
        fields = [value.strip() for value in row]
        if len(fields) != 2 or not fields[1].isdigit():
            continue
        processes.append((fields[0], int(fields[1])))

    owners = query_process_owners({pid for _, pid in processes})
    index_by_uuid = {summary.uuid: summary.index for summary in summaries}
    process_summaries: dict[int, ProcessSummary] = {}
    for uuid, pid in processes:
        if uuid not in index_by_uuid:
            continue
        process_summary = process_summaries.setdefault(
            index_by_uuid[uuid], ProcessSummary()
        )
        process_summary.count += 1
        process_summary.owners.add(owners.get(pid, "unknown"))
    return process_summaries


def occupancy_status(
    summary: GpuSummary,
    processes: ProcessSummary,
    active_threshold: float,
    busy_threshold: float,
) -> str:
    if summary.average_utilization >= busy_threshold:
        return "BUSY"
    if summary.average_utilization >= active_threshold:
        return "ACTIVE"
    if processes.count or summary.max_memory_used >= 1024:
        return "RESERVED"
    return "IDLE"


def find_visible_gpu(
    summaries: list[GpuSummary], visible_devices: str
) -> GpuSummary | None:
    if "," in visible_devices:
        return None
    for summary in summaries:
        if visible_devices in {str(summary.index), summary.uuid}:
            return summary
    return None


def print_summary(
    summaries: list[GpuSummary],
    process_summaries: dict[int, ProcessSummary],
    selected_index: int | None,
    active_threshold: float,
    busy_threshold: float,
) -> None:
    print("GPU occupancy (* = selected/preallocated physical GPU):")
    print("    GPU  STATE       AVG%  PEAK%     MEMORY GiB  PROCS  OWNERS")
    for summary in summaries:
        processes = process_summaries.get(summary.index, ProcessSummary())
        status = occupancy_status(summary, processes, active_threshold, busy_threshold)
        owner_names = sorted(processes.owners)
        owners = ",".join(owner_names[:4]) or "-"
        if len(owner_names) > 4:
            owners += f",+{len(owner_names) - 4} more"
        marker = "*" if summary.index == selected_index else " "
        memory = (
            f"{summary.max_memory_used / 1024:.1f}/{summary.memory_total / 1024:.1f}"
        )
        print(
            f" {marker}  {summary.index:>3}  {status:<9} "
            f"{summary.average_utilization:>5.1f} "
            f"{summary.peak_utilization:>6.1f} "
            f"{memory:>14} {processes.count:>6}  {owners}"
        )
    print(
        f"States: BUSY >= {busy_threshold:g}%; ACTIVE >= {active_threshold:g}%; "
        "RESERVED = lower load but holding memory/processes; otherwise IDLE."
    )


def write_environment(path: Path, values: dict[str, str]) -> None:
    content = "".join(
        f"export {name}={shlex.quote(value)}\n" for name, value in values.items()
    )
    path.write_text(content, encoding="utf-8")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize NVIDIA GPU occupancy and select a CI device."
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        required=True,
        help="write shell exports for the calling CI job to this file",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--min-free-mib", type=int, default=4096)
    parser.add_argument("--active-threshold", type=float, default=10.0)
    parser.add_argument("--busy-threshold", type=float, default=80.0)
    return parser.parse_args()


def main() -> int:
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_arguments()
    if args.samples < 1 or args.interval < 0:
        raise PreflightError("Sample count must be positive and interval non-negative")
    if not 0 <= args.active_threshold < args.busy_threshold <= 100:
        raise PreflightError(
            "Occupancy thresholds must satisfy 0 <= active < busy <= 100"
        )

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    requested_gpu = os.environ.get("GPU_DEVICE", "").strip()
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"=== GPU CI preflight | {socket.getfqdn()} | {timestamp} ===")
    print(
        f"Initial allocation: CUDA_VISIBLE_DEVICES={visible_devices or '<unset>'}; "
        f"GPU_DEVICE={requested_gpu or '<unset>'}"
    )

    samples = collect_samples(args.samples, args.interval)
    summaries = summarize_samples(samples)
    process_summaries = query_processes(summaries)

    if visible_devices:
        selected = find_visible_gpu(summaries, visible_devices)
        environment = {
            "CUDA_VISIBLE_DEVICES": visible_devices,
            "GPU_DEVICE": requested_gpu or "0",
        }
        decision = f"preserve preallocated CUDA_VISIBLE_DEVICES={visible_devices}"
    else:
        selected = select_gpu(summaries, requested_gpu, args.min_free_mib)
        environment = {
            "DARTS_CI_PHYSICAL_GPU_DEVICE": str(selected.index),
            "CUDA_VISIBLE_DEVICES": str(selected.index),
            "GPU_DEVICE": "0",
        }
        reason = "explicit GPU_DEVICE override" if requested_gpu else "lowest load"
        decision = f"physical GPU {selected.index} ({reason}) -> logical CUDA device 0"

    selected_index = selected.index if selected is not None else None
    print_summary(
        summaries,
        process_summaries,
        selected_index,
        args.active_threshold,
        args.busy_threshold,
    )
    print(f"Decision: {decision}")
    if selected is not None and selected.average_utilization >= args.busy_threshold:
        print(
            f"WARNING: selected GPU {selected.index} is BUSY at "
            f"{selected.average_utilization:.1f}% average utilization",
        )

    write_environment(args.env_file, environment)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PreflightError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1) from error
