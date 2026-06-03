#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 ETH Zurich and University of Bologna
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper that invokes the shared Deeploy test runner for the XDNA2 platform.

Usage (from DeeployTest/):
    python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular [--skipsim] [-v]
    python deeployRunner_xdna2.py -t Tests/Kernels/BF16/Add/Regular --trace [--trace-buffer-size 16384]
"""

import json
import os
import shutil
import subprocess
import sys
from glob import glob

from testUtils.deeployRunner import main


def _add_xdna2_args(parser):
    """Register XDNA2-specific CLI arguments."""
    parser.add_argument('--trace',
                        action = 'store_true',
                        default = False,
                        help = 'Enable execution tracing in the generated MLIR')
    parser.add_argument('--trace-buffer-size',
                        type = int,
                        default = 8192,
                        help = 'Trace buffer size in bytes (default: 8192)')
    parser.add_argument('--trace-shim-col',
                        type = int,
                        default = 7,
                        help = 'Shim column to egress trace packets through (default: 7). ')
    parser.add_argument('--trace-tiles',
                        type = str,
                        default = None,
                        help = 'Comma-separated tile selector for tracing, e.g. "c0r2,c1r2". '
                        'Only meaningful with --trace. If omitted, every active core is traced.')
    parser.add_argument('--analyzeTrace',
                        action = 'store_true',
                        default = False,
                        help = 'After simulation, run scripts/trace_boundness.py on the '
                        'parsed trace.json to report per-core memory-boundness. '
                        'Requires --trace.')
    parser.add_argument('--visualize-routing',
                        action = 'store_true',
                        default = False,
                        help = 'After codegen, dump per-flow ASCII switchbox-routing diagrams.')
    parser.add_argument('--num-col',
                        type = int,
                        default = 1,
                        help = 'Number of AIE columns to use (1 to 8, default: 1).')
    parser.add_argument('--num-aie-row',
                        type = int,
                        default = 1,
                        help = 'Number of AIE compute rows per column to use (1 to 4, default: 1). '
                        'Total active AIE compute tiles = num-col * num-aie-row.')
    parser.add_argument('--data-mode',
                        type = str,
                        choices = ['auto', 'embed', 'file'],
                        default = 'auto',
                        help = 'Where test inputs/outputs live: embed in header, .bin sidecars, '
                        'or auto-pick by size (default: auto).')


def _add_xdna2_gen_args(args, gen_args_list):
    """Forward XDNA2-specific arguments to the generation script."""
    if getattr(args, 'trace', False):
        gen_args_list.append('--trace')
        trace_buffer_size = getattr(args, 'trace_buffer_size', 8192)
        if trace_buffer_size != 8192:
            gen_args_list.append(f'--trace-buffer-size={trace_buffer_size}')
        trace_shim_col = getattr(args, 'trace_shim_col', 7)
        if trace_shim_col != 7:
            gen_args_list.append(f'--trace-shim-col={trace_shim_col}')
        trace_tiles = getattr(args, 'trace_tiles', None)
        if trace_tiles:
            gen_args_list.append(f'--trace-tiles={trace_tiles}')
    num_col = int(getattr(args, 'num_col', 1) or 1)
    num_aie_row = int(getattr(args, 'num_aie_row', 1) or 1)
    if num_col > 1:
        gen_args_list.append(f'--num-col={num_col}')
    if num_aie_row > 1:
        gen_args_list.append(f'--num-aie-row={num_aie_row}')
    data_mode = getattr(args, 'data_mode', 'auto')
    if data_mode != 'auto':
        gen_args_list.append(f'--data-mode={data_mode}')


def _xdna2_visualize_routing(config, args):
    """Render per-flow switchbox routing as ASCII diagrams under <gen_dir>/routes/.

    Pipeline:
      1. aie-opt lowers ObjectFifos to flows and resolves switchbox routes.
      2. aie-translate dumps the resolved switchboxes + routes to JSON.
      3. The vendored aie_visualize_routing.py renders one route*.txt per flow.
    """
    if not getattr(args, 'visualize_routing', False):
        return

    network_mlir = os.path.join(config.gen_dir, "network.mlir")
    if not os.path.isfile(network_mlir):
        print(f"Warning: --visualize-routing set but {network_mlir} not found; skipping.")
        return

    try:
        import aie.utils.config as aie_cfg
        aie_bin = os.path.join(aie_cfg.root_path(), "bin")
    except Exception as e:
        print(f"Warning: could not locate aie-opt / aie-translate ({e}); skipping routing visualization.")
        return

    aie_opt = os.path.join(aie_bin, "aie-opt")
    aie_translate = os.path.join(aie_bin, "aie-translate")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    visualize_py = os.path.join(repo_root, "scripts", "aie_visualize_routing.py")
    
    if not all(os.path.isfile(p) for p in (aie_opt, aie_translate, visualize_py)):
        missing = [p for p in (aie_opt, aie_translate, visualize_py) if not os.path.isfile(p)]
        print(f"Warning: routing-viz dependencies missing: {missing}; skipping.")
        return

    flows_json = os.path.join(config.gen_dir, "flows.json")
    routes_dir = os.path.join(config.gen_dir, "routes")
    
    if os.path.isdir(routes_dir):
        shutil.rmtree(routes_dir)
    os.makedirs(routes_dir)

    try:
        opt = subprocess.run(
            [aie_opt,
             "--aie-objectFifo-stateful-transform",
             "--aie-create-pathfinder-flows",
             "--aie-find-flows",
             network_mlir],
            check = True, capture_output = True)
        translate = subprocess.run(
            [aie_translate, "--aie-flows-to-json"],
            input = opt.stdout, check = True, capture_output = True)
        with open(flows_json, "wb") as f:
            f.write(translate.stdout)
    except subprocess.CalledProcessError as e:
        print(f"Warning: routing-viz aie-opt/aie-translate failed: "
              f"{e.stderr.decode(errors='replace') if e.stderr else e}")
        return

    try:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf8")
        subprocess.run(
            [sys.executable, visualize_py, "-j", flows_json, "-o", routes_dir],
            check = True, capture_output = True, env = env)
    except subprocess.CalledProcessError as e:
        print(f"Warning: routing-viz renderer failed: "
              f"{e.stderr.decode(errors='replace') if e.stderr else e}")
        return

    n_routes = len([f for f in os.listdir(routes_dir) if f.startswith("route") and f.endswith(".txt")])
    print(f"Routing visualization: {n_routes} route diagrams under {routes_dir}/")


def _run_trace_boundness_analyzer(trace_json_path):
    """Invoke scripts/trace_boundness.py on a parsed trace.json.

    Runs as a subprocess so the analyzer's stdlib-only invariant stays
    intact (no import dance against Deeploy/mlir-aie internals). Prints
    the report inline so it's visible in the standard test runner log.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    analyzer = os.path.join(here, "..", "scripts", "trace_boundness.py")
    analyzer = os.path.normpath(analyzer)
    if not os.path.isfile(analyzer):
        print(f"Warning: --analyzeTrace set but analyzer not found at {analyzer}; skipping.")
        return
    try:
        result = subprocess.run([sys.executable, analyzer, "-i", trace_json_path],
                                check = True, capture_output = True, text = True)
        print("\n=== Trace boundness analysis ===")
        print(result.stdout, end = "")
    except subprocess.CalledProcessError as e:
        print(f"Warning: trace boundness analyzer failed: "
              f"{e.stderr if e.stderr else e}")


def _xdna2_post_sim(config, result, args):
    """Parse trace.txt into a Perfetto-compatible trace.json after simulation.

    Both trace.txt and trace.json are also copied into config.gen_dir (the
    per-test generated-source directory) so the trace artifacts live next to
    the network.onnx / network.mlir they describe — easier to archive,
    compare across runs, and gitignore than the build-tree copies.

    Also runs routing visualization if requested. That step doesn't depend
    on simulation output and could in principle run earlier, but post_sim
    is the only callback the shared runner exposes.
    """
    _xdna2_visualize_routing(config, args)

    if not getattr(args, 'trace', False):
        if getattr(args, 'analyzeTrace', False):
            print("Warning: --analyzeTrace requires --trace; ignoring.")
        return

    build_dir = config.build_dir
    trace_txt = os.path.join(build_dir, "bin", "trace.txt")
    if not os.path.isfile(trace_txt):
        print(f"Warning: --trace enabled but {trace_txt} not found; skipping trace parsing.")
        return

    # Find the MLIR with lowered NpuWrite32 ops (trace event register config).
    # aiecc.py produces this when invoked with --dump-intermediates.
    prj_pattern = os.path.join(build_dir, "DeeployTest", "Platforms", "XDNA2", "network.mlir.prj",
                               "main_physical_with_elfs.mlir")
    candidates = glob(prj_pattern)
    if not candidates:
        print(f"Warning: lowered MLIR not found at {prj_pattern}; skipping trace parsing.")
        return
    lowered_mlir = candidates[0]

    trace_json = os.path.join(build_dir, "bin", "trace.json")

    # Mirror the raw trace.txt next to the test artifacts immediately so
    # it's preserved even if mlir-aie's parser bails on the .json conversion.
    try:
        os.makedirs(config.gen_dir, exist_ok = True)
        shutil.copy2(trace_txt, os.path.join(config.gen_dir, "trace.txt"))
    except Exception as e:
        print(f"Warning: could not copy trace.txt to {config.gen_dir}: {e}")

    try:
        from aie.utils.trace.parse import align_column_start_index, check_for_valid_trace, convert_commands_to_json, \
            convert_to_byte_stream, convert_to_commands, parse_mlir_trace_events, setup_trace_metadata, \
            trace_pkts_de_interleave, trim_trace_pkts

        with open(trace_txt, "r") as f:
            trace_pkts = f.read().split("\n")

        with open(lowered_mlir, "r") as f:
            mlir_str = f.read()

        pid_events, events_module = parse_mlir_trace_events(mlir_str)

        if not check_for_valid_trace(trace_txt, trace_pkts):
            print(f"Warning: trace data in {trace_txt} appears invalid; skipping trace parsing.")
            return

        trimmed = trim_trace_pkts(trace_pkts)
        sorted_pkts = trace_pkts_de_interleave(trimmed)
        byte_streams = convert_to_byte_stream(sorted_pkts)
        commands = convert_to_commands(byte_streams, False)

        # Long-running kernels overflow the 8 KB trace buffer many times,
        # leaving us mid-stream after the final wrap. The first decoded
        # command can be a `Repeat`, which references the previous event's
        # `cycles` and `multiple_list` — locals that aren't yet bound in
        # mlir-aie's parser, raising an UnboundLocalError. Drop leading
        # Repeats so parsing starts from the first concrete event.
        for tt_cmds in commands:
            if not isinstance(tt_cmds, dict):
                continue
            for loc, cmd_list in list(tt_cmds.items()):
                drop = 0
                while drop < len(cmd_list) and "Repeat" in cmd_list[drop].get("type", ""):
                    drop += 1
                if drop:
                    tt_cmds[loc] = cmd_list[drop:]

        pid_events = align_column_start_index(pid_events, commands)

        trace_events = []
        setup_trace_metadata(trace_events, pid_events, events_module)
        convert_commands_to_json(trace_events, commands, pid_events, events_module)

        with open(trace_json, "w") as f:
            json.dump(trace_events, f)

        print(f"Trace parsed: {trace_json} ({len(trace_events)} events)")

        # Mirror the parsed Perfetto JSON next to the test artifacts as well.
        try:
            shutil.copy2(trace_json, os.path.join(config.gen_dir, "trace.json"))
            print(f"Trace artifacts copied to {config.gen_dir}/{{trace.txt, trace.json}}")
        except Exception as e:
            print(f"Warning: could not copy trace.json to {config.gen_dir}: {e}")

        if getattr(args, 'analyzeTrace', False):
            _run_trace_boundness_analyzer(trace_json)

    except SystemExit:
        print(f"Warning: trace parsing failed (mlir-aie parser error). "
              f"Ensure the build was done with --trace enabled.")
    except Exception as e:
        print(f"Warning: trace parsing failed: {e}")


if __name__ == '__main__':
    sys.exit(
        main(default_platform = "XDNA2",
             default_simulator = "host",
             tiling_enabled = True,
             platform_specific_cmake_args = [f"-DPython3_EXECUTABLE={sys.executable}"],
             parser_setup_callback = _add_xdna2_args,
             gen_args_callback = _add_xdna2_gen_args,
             post_sim_callback = _xdna2_post_sim))
