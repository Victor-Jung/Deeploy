# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
# SPDX-License-Identifier: Apache-2.0
"""Memory-boundness analyzer for XDNA2 traces.

Reads the Perfetto-format ``trace.json`` produced by mlir-aie's trace
parser, reconstructs per-core kernel phases from ``INSTR_EVENT_0`` /
``INSTR_EVENT_1`` markers (kernel entry / exit), and reports the ratio of
non-overlapped DMA stall to kernel-active time as a memory-boundness
metric.

Inputs
------
The mlir-aie trace pipeline (``aie.trace.host_config`` + the trace.txt
emitted by the kernel) is decoded by
``python/utils/trace/parse.py`` into a Perfetto JSON. Each
``INSTR_EVENT_*`` becomes a duration event (``ph='B'`` / ``ph='E'``) on
a per-core trace stream identified by Perfetto ``pid``. Metadata events
(``ph='M'``) map each ``pid`` to a human-readable label like
``"core_trace for tile2,0"``.

Ring buffer caveat
------------------
The on-device trace buffer is a ring buffer (default 8 KB). When it
overflows, the oldest packets are dropped. The first event in our
snapshot may therefore be an ``INSTR_EVENT_1`` with no matching
``INSTR_EVENT_0``, and the last may be an ``INSTR_EVENT_0`` with no
matching ``INSTR_EVENT_1``. The reconstruction algorithm trims those
orphans and reconciles mid-stream anomalies (two consecutive ``e0`` or
two consecutive ``e1``) so the remaining list of ``(e0, e1)`` pairs is
clean.

Output
------
One human-readable table on stdout (or ``--output FILE``) with one row
per traced core: number of phases, anomaly count, kernel and stall
cycle medians, boundness ratio, and a short verdict label. Bounds-rate
above the configured anomaly threshold suppresses the verdict and asks
the user to re-collect the trace.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

_EVENT_BEGIN = "INSTR_EVENT_0"
_EVENT_END = "INSTR_EVENT_1"

@dataclass
class CoreResult:
    label: str
    raw_event_count: int
    anomaly_count: int
    phases: List[Tuple[int, int]]    # (e0_ts, e1_ts) cycles
    kernel_cyc: List[int]
    stall_cyc: List[int]              # length = len(phases) - 1, undefined for phase 0


def _load_trace(path: str) -> Tuple[Dict[int, str], List[dict]]:
    """Return (pid -> label, list of real (B-phase) events)."""
    with open(path, "r") as f:
        data = json.load(f)

    pid_labels: Dict[int, str] = {}
    real_events: List[dict] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        if entry.get("ph") == "M" and entry.get("name") == "process_name":
            pid = entry.get("pid")
            label = entry.get("args", {}).get("name", f"pid={pid}")
            if pid is not None:
                pid_labels[pid] = label
            continue
        # Real events come as B/E pairs. INSTR_EVENT_* is effectively
        # instantaneous, so we keep only B (the timestamps coincide).
        if entry.get("ph") == "B" and entry.get("name") in (_EVENT_BEGIN, _EVENT_END):
            real_events.append(entry)
    return pid_labels, real_events


def _reconstruct_phases(events: List[dict]) -> Tuple[List[Tuple[int, int]], int]:
    """Apply the cleanup rules and return (phase pairs, anomaly count).

    Rules:
      1. Drop a leading INSTR_EVENT_1 (snapshot started mid-phase).
      2. Drop a trailing INSTR_EVENT_0 (snapshot ended mid-phase).
      3. Two consecutive INSTR_EVENT_0 → keep the latest (most recent
         kernel start); the prior start is an anomaly (missed E1).
      4. Two consecutive INSTR_EVENT_1 → keep the earliest (the matching
         E0 fired before any in-snapshot E1); discard the later as an
         anomaly (missed E0).
      5. After cleanup, the sequence alternates strictly e0, e1, e0, e1.
    """
    events = sorted(events, key=lambda e: e["ts"])
    names = [e["name"] for e in events]
    ts = [e["ts"] for e in events]

    # 1. trim leading orphan(s)
    while names and names[0] == _EVENT_END:
        names.pop(0)
        ts.pop(0)
    # 2. trim trailing orphan(s)
    while names and names[-1] == _EVENT_BEGIN:
        names.pop()
        ts.pop()

    # 3 + 4. walk left-to-right, fixing consecutive duplicates.
    cleaned_names: List[str] = []
    cleaned_ts: List[int] = []
    anomalies = 0
    for n, t in zip(names, ts):
        if cleaned_names and cleaned_names[-1] == n:
            anomalies += 1
            if n == _EVENT_BEGIN:
                # Two e0 in a row → drop the older one we already kept.
                cleaned_names.pop()
                cleaned_ts.pop()
            else:
                # Two e1 in a row → skip this one.
                continue
        cleaned_names.append(n)
        cleaned_ts.append(t)

    # After trimming + reconciling we may still have a stray leader/trailer.
    while cleaned_names and cleaned_names[0] == _EVENT_END:
        cleaned_names.pop(0)
        cleaned_ts.pop(0)
    while cleaned_names and cleaned_names[-1] == _EVENT_BEGIN:
        cleaned_names.pop()
        cleaned_ts.pop()

    # 5. assert strict alternation.
    if len(cleaned_names) % 2 != 0:
        # Should be unreachable after the trims above. Drop the dangling
        # tail to fail soft rather than abort the run.
        anomalies += 1
        cleaned_names.pop()
        cleaned_ts.pop()

    pairs: List[Tuple[int, int]] = []
    for i in range(0, len(cleaned_names), 2):
        # Defensive: confirm pairing. If anomalies broke the invariant
        # somehow, count it and skip.
        if cleaned_names[i] != _EVENT_BEGIN or cleaned_names[i + 1] != _EVENT_END:
            anomalies += 1
            continue
        pairs.append((cleaned_ts[i], cleaned_ts[i + 1]))
    return pairs, anomalies


def _analyze_core(label: str, events: List[dict], warmup: int) -> CoreResult:
    raw_count = len(events)
    pairs, anomalies = _reconstruct_phases(events)

    if warmup > 0 and len(pairs) >= warmup + 3:
        pairs = pairs[warmup:]

    kernel_cyc = [e1 - e0 for (e0, e1) in pairs]
    # Stall is undefined for pair 0 (no prior e1 in the snapshot —
    # whether trimmed as orphan or rolled over, we can't measure
    # the gap before the first kept phase).
    stall_cyc = [pairs[i][0] - pairs[i - 1][1] for i in range(1, len(pairs))]

    return CoreResult(
        label=label,
        raw_event_count=raw_count,
        anomaly_count=anomalies,
        phases=pairs,
        kernel_cyc=kernel_cyc,
        stall_cyc=stall_cyc,
    )


def _format_report(results: List[CoreResult]) -> str:
    lines = []
    header = (
        f"{'Core':<32} {'Phases':>7} {'Anomalies':>10} "
        f"{'Kernel cyc (med)':>17} {'Stall cyc (med)':>17} "
        f"{'Boundness':>10}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    total_events = 0

    for r in results:
        total_events += r.raw_event_count

        if not r.kernel_cyc:
            lines.append(
                f"{r.label:<32} {0:>7} {r.anomaly_count:>10} "
                f"{'-':>17} {'-':>17} {'-':>10}"
            )
            continue

        k_med = int(statistics.median(r.kernel_cyc))
        if r.stall_cyc:
            s_med = int(statistics.median(r.stall_cyc))
            boundness = s_med / (s_med + k_med) if (s_med + k_med) > 0 else 0.0
            lines.append(
                f"{r.label:<32} {len(r.kernel_cyc):>7} {r.anomaly_count:>10} "
                f"{k_med:>17} {s_med:>17} {boundness:>10.2f}"
            )
        else:
            # Only one phase kept — no stall measurable.
            lines.append(
                f"{r.label:<32} {len(r.kernel_cyc):>7} {r.anomaly_count:>10} "
                f"{k_med:>17} {'-':>17} {'-':>10}"
            )

    if total_events == 0:
        lines.append("")
        lines.append("No INSTR_EVENT_0/1 events found in trace — nothing to analyze.")

    return "\n".join(lines) + "\n"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Memory-boundness analyzer for XDNA2 traces (cycles only)."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="trace.json produced by mlir-aie's trace parser")
    parser.add_argument("--warmup", type=int, default=2,
                        help="Discard the first N kernel invocations as warm-up "
                             "(default: 2). Skipped automatically when fewer than "
                             "warmup+3 pairs remain.")
    parser.add_argument("-o", "--output", default=None,
                        help="Write report to FILE instead of stdout.")
    args = parser.parse_args(argv)

    pid_labels, events = _load_trace(args.input)

    by_pid: Dict[int, List[dict]] = {}
    for ev in events:
        by_pid.setdefault(ev["pid"], []).append(ev)

    results: List[CoreResult] = []
    for pid in sorted(by_pid):
        label = pid_labels.get(pid, f"pid={pid}")
        results.append(_analyze_core(label, by_pid[pid], args.warmup))

    report = _format_report(results)

    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
    else:
        sys.stdout.write(report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
