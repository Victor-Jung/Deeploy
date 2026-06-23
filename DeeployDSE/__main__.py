# SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""DeeployDSE CLI: sweep design space for XDNA2 operators.

Usage:
    # Run a sweep
    python -m DeeployDSE sweep -t DeeployTest/Tests/Kernels/BF16/Add/Regular
    python -m DeeployDSE sweep -t DeeployTest/Tests/Kernels/BF16/Add/Regular --cols 1,2,4,8 --rows 1,2

    # Regenerate plots from existing CSV (after manual edits)
    python -m DeeployDSE plot -o dse_results
    python -m DeeployDSE plot -o dse_results --peak-bw 57.6
"""

import argparse
import os
import sys

from DeeployDSE.analyze import load_csv, plot_pareto, plot_roofline, save_csv
from DeeployDSE.generator import XDNA2ElementwiseGenerator
from DeeployDSE.runner import DSERunner


def _do_plot(output_dir: str, peak_bw: float):
    csv_path = os.path.join(output_dir, "results.csv")
    if not os.path.isfile(csv_path):
        print(f"No results.csv found in {output_dir}")
        sys.exit(1)
    all_results = load_csv(csv_path)
    plot_roofline(all_results, peak_bw_gbps = peak_bw, output_path = os.path.join(output_dir, "roofline.html"))
    ops = sorted(set(r.op for r in all_results if r.passed))
    for op in ops:
        plot_pareto(all_results, op = op, output_path = os.path.join(output_dir, f"pareto_{op}.html"))
    print(f"Plots written to {output_dir}/ (roofline + {len(ops)} pareto)")


def main():
    parser = argparse.ArgumentParser(description = "DeeployDSE: Design Space Exploration for XDNA2")
    subparsers = parser.add_subparsers(dest = "command")

    # -- sweep subcommand --
    sweep_p = subparsers.add_parser("sweep", help = "Run a DSE sweep on hardware")
    sweep_p.add_argument("--test-dir", "-t", required = True, help = "Path to test directory")
    sweep_p.add_argument("--cols", default = None, help = "Comma-separated column counts (default: 1-8)")
    sweep_p.add_argument("--rows", default = None, help = "Comma-separated row counts (default: 1-4)")
    sweep_p.add_argument("--output", "-o", default = "dse_results", help = "Output directory")
    sweep_p.add_argument("--peak-bw", type = float, default = 57.6, help = "Peak bandwidth GB/s for roofline")
    sweep_p.add_argument("--no-plot", action = "store_true", help = "Skip plotting")

    # -- plot subcommand --
    plot_p = subparsers.add_parser("plot", help = "Regenerate plots from results.csv")
    plot_p.add_argument("--output", "-o", default = "dse_results", help = "Directory containing results.csv")
    plot_p.add_argument("--peak-bw", type = float, default = 57.6, help = "Peak bandwidth GB/s for roofline")

    args = parser.parse_args()

    if args.command == "plot":
        _do_plot(args.output, args.peak_bw)
        return

    if args.command != "sweep":
        parser.print_help()
        sys.exit(1)

    col_range = [int(x) for x in args.cols.split(",")] if args.cols else None
    row_range = [int(x) for x in args.rows.split(",")] if args.rows else None

    generator = XDNA2ElementwiseGenerator(col_range = col_range, row_range = row_range)
    configs = generator.generate(args.test_dir)

    if not configs:
        print("No legal configurations found.")
        sys.exit(1)

    print(f"Generated {len(configs)} design points for {configs[0].op}")
    runner = DSERunner()
    results = []

    for i, cfg in enumerate(configs):
        print(f"  [{i+1}/{len(configs)}] {cfg.label} ...", end = " ", flush = True)
        result = runner.run(cfg)
        status = "PASS" if result.passed else "FAIL"
        lat = f"{result.latency.median_us:.1f}μs" if result.latency.median_us > 0 else "N/A"
        print(f"{status} {lat} {result.throughput_gbps:.2f} GB/s")
        results.append(result)

    # Append to shared CSV
    os.makedirs(args.output, exist_ok = True)
    csv_path = os.path.join(args.output, "results.csv")
    save_csv(results, csv_path, append = True)
    print(f"\nResults appended to {csv_path}")

    # Plot from full aggregated CSV
    if not args.no_plot:
        try:
            _do_plot(args.output, args.peak_bw)
        except ImportError as e:
            print(f"Plotting skipped: {e}")


if __name__ == "__main__":
    main()
