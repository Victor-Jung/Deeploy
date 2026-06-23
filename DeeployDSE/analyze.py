# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

from DeeployDSE.results import DSEResult, LatencyStats

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    go = None


def results_to_records(results: List[DSEResult]) -> List[dict]:
    """Flatten results into dicts suitable for CSV/DataFrame."""
    records = []
    for r in results:
        records.append({
            "op": r.op,
            "num_col": r.num_col,
            "num_aie_row": r.num_aie_row,
            "total_cores": r.total_cores,
            "label": r.label,
            "passed": r.passed,
            "errors": r.errors,
            "total_elems": r.total_elems,
            "latency_min_us": r.latency.min_us,
            "latency_median_us": r.latency.median_us,
            "latency_mean_us": r.latency.mean_us,
            "latency_max_us": r.latency.max_us,
            "latency_stdev_us": r.latency.stdev_us,
            "throughput_gbps": r.throughput_gbps,
            "total_ops": r.total_ops,
            "total_bytes": r.total_bytes,
        })
    return records


def save_csv(results: List[DSEResult], path: str, append: bool = False) -> None:
    """Write results to CSV. If append=True, add rows to existing file."""
    import csv
    records = results_to_records(results)
    if not records:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    file_exists = os.path.isfile(path) and append
    mode = "a" if append else "w"
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=records[0].keys())
        if not file_exists:
            w.writeheader()
        w.writerows(records)


def load_csv(path: str) -> List[DSEResult]:
    """Load results from a CSV file."""
    import csv
    results = []
    with open(path) as f:
        for row in csv.DictReader(f):
            results.append(DSEResult(
                op=row['op'],
                num_col=int(row['num_col']),
                num_aie_row=int(row['num_aie_row']),
                label=row['label'],
                passed=row['passed'] == 'True',
                errors=int(row['errors']),
                total_elems=int(row['total_elems']),
                latency=LatencyStats(
                    min_us=float(row['latency_min_us']),
                    median_us=float(row['latency_median_us']),
                    mean_us=float(row['latency_mean_us']),
                    max_us=float(row['latency_max_us']),
                    stdev_us=float(row['latency_stdev_us']),
                ),
                throughput_gbps=float(row['throughput_gbps']),
                total_ops=int(row.get('total_ops', 0)),
                input_bytes=int(row['total_bytes']) // 2,
                output_bytes=int(row['total_bytes']) - int(row['total_bytes']) // 2,
                build_success=True,
            ))
    return results


def plot_roofline(results: List[DSEResult],
                  peak_bw_gbps: float = 57.6,
                  peak_gflops: float = None,
                  freq_ghz: float = 1.8,
                  macs_per_tile: int = 16,
                  num_tiles: int = 32,
                  output_path: Optional[str] = None) -> None:
    """Classical roofline: attained GFLOP/s vs operational intensity (FLOP/Byte).

    Peak compute = num_tiles * macs_per_tile * 2 * freq_ghz (GFLOP/s).
    Operational intensity = total_flops / total_bytes_moved.
    Attained perf = total_flops / median_latency.
    """
    if go is None:
        raise ImportError("plotly required for plotting: pip install plotly")

    if peak_gflops is None:
        peak_gflops = num_tiles * macs_per_tile * 2 * freq_ghz  # 2 ops per MAC

    passed = [r for r in results if r.passed and r.latency.median_us > 0]
    if not passed:
        print("No passing results to plot.")
        return

    fig = go.Figure()

    # Group by op for coloring
    ops = sorted(set(r.op for r in passed))
    for op in ops:
        op_results = [r for r in passed if r.op == op]
        xs, ys, labels = [], [], []
        for r in op_results:
            total_flops = r.total_ops if r.total_ops > 0 else r.total_elems
            oi = total_flops / r.total_bytes
            attained = (total_flops / r.latency.median_us) * 1e-3
            xs.append(oi)
            ys.append(attained)
            labels.append(r.label)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="markers+text", text=labels,
            textposition="top center", marker=dict(size=10),
            name=op,
        ))

    # Roofline ceilings
    oi_range = np.logspace(-2, 2, 200)
    ridge_point = peak_gflops / peak_bw_gbps  # FLOP/Byte where BW meets compute
    roofline = np.minimum(peak_gflops, peak_bw_gbps * oi_range)

    fig.add_trace(go.Scatter(
        x=oi_range.tolist(), y=roofline.tolist(), mode="lines",
        line=dict(color="red", dash="dash"), name="Roofline",
    ))

    fig.update_layout(
        title=f"Roofline (peak: {peak_gflops:.0f} GFLOP/s, BW: {peak_bw_gbps} GB/s)",
        xaxis_title="Operational Intensity (FLOP/Byte)",
        yaxis_title="Attained Performance (GFLOP/s)",
        xaxis=dict(type="log", dtick=1, exponentformat="power",
                   minor=dict(dtick="D1", showgrid=True)),
        yaxis=dict(type="log", dtick=1, exponentformat="power",
                   minor=dict(dtick="D1", showgrid=True)),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_html(output_path)
    else:
        fig.show()


def plot_pareto(results: List[DSEResult], op: Optional[str] = None,
                output_path: Optional[str] = None) -> None:
    """Pareto front: latency vs resource usage (cores). Filters by op if given."""
    if go is None:
        raise ImportError("plotly required for plotting: pip install plotly")

    passed = [r for r in results if r.passed]
    if op:
        passed = [r for r in passed if r.op == op]
    if not passed:
        print(f"No passing results to plot{f' for {op}' if op else ''}.")
        return

    cores = np.array([r.total_cores for r in passed])
    latency = np.array([r.latency.median_us for r in passed])
    labels = [r.label for r in passed]

    # Compute Pareto front (minimize both latency and cores)
    pareto_mask = _pareto_front(np.column_stack([latency, cores]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cores, y=latency, mode="markers+text", text=labels,
        textposition="top center", marker=dict(size=8, color="gray"),
        name="All points",
    ))

    pareto_idx = np.where(pareto_mask)[0]
    # Sort Pareto points by cores for line
    order = np.argsort(cores[pareto_idx])
    fig.add_trace(go.Scatter(
        x=cores[pareto_idx][order], y=latency[pareto_idx][order],
        mode="markers+lines",
        marker=dict(size=12, color="red", symbol="star"),
        name="Pareto front",
    ))

    fig.update_layout(
        title=f"Pareto Front: Latency vs Resource Usage{f' ({op})' if op else ''}",
        xaxis_title="Total AIE cores",
        yaxis_title="Median latency (μs)",
        xaxis=dict(dtick=1),
    )

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.write_html(output_path)
    else:
        fig.show()


def _pareto_front(costs: np.ndarray) -> np.ndarray:
    """Boolean mask of Pareto-optimal rows (minimize all objectives)."""
    n = len(costs)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        # A point is dominated if another point is <= on all objectives and < on at least one
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            if np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i]):
                is_pareto[i] = False
                break
    return is_pareto
