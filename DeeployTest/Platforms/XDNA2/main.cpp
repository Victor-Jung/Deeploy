// SPDX-FileCopyrightText: 2026 ETH Zurich and University of Bologna
//
// SPDX-License-Identifier: Apache-2.0

// XRT C++ testbench for the XDNA2 (AIE2p) platform.
// Loads network.xclbin produced by aiecc.py, runs the MLIR_AIE kernel,
// reads back outputs and compares against golden reference values.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"

// File mode is used automatically for tensors >32 MiB so they don't have to
// be embedded as C++ literals (which OOMs the compiler on large tests).
#include "testinputs.h"
#include "testoutputs.h"

// ---------------------------------------------------------------------------
// BF16 helpers
// ---------------------------------------------------------------------------
static float bf16_to_float(uint16_t bf16) {
  uint32_t f32_bits = static_cast<uint32_t>(bf16) << 16;
  float f;
  std::memcpy(&f, &f32_bits, sizeof(f));
  return f;
}

static bool bf16_nearly_equal(uint16_t a, uint16_t b,
                              unsigned int tolerance_ulps = 1,
                              float rtol = 0.0f, float atol = 0.0f) {
  float fa = bf16_to_float(a);
  float fb = bf16_to_float(b);
  float diff = std::fabs(fa - fb);

  uint16_t ref_exp = (b >> 7) & 0xFF;
  float ulp;
  if (ref_exp == 0)
    ulp = std::ldexp(1.0f, -133);
  else
    ulp = std::ldexp(1.0f, static_cast<int>(ref_exp) - 127 - 7);

  float tol = std::fmax(atol + rtol * std::fabs(fb), ulp * tolerance_ulps);
  return diff <= tol;
}

// ---------------------------------------------------------------------------
// Read the NPU instruction binary produced by aiecc.py
// ---------------------------------------------------------------------------
static std::vector<uint32_t> read_instr_binary(const std::string &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open instruction file: " + path);
  }
  file.seekg(0, std::ios::end);
  size_t byte_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint32_t> instr(byte_size / sizeof(uint32_t));
  file.read(reinterpret_cast<char *>(instr.data()), byte_size);
  return instr;
}

// ---------------------------------------------------------------------------
// Test data loader. Used in file mode for large tensors that
// can't be embedded as C++ literals. The .bin files live next to the
// binary and are raw uint16 BF16 dumps.
// ---------------------------------------------------------------------------
static std::vector<uint16_t> read_bf16_binary(const std::string &path,
                                              size_t expected_elems) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open data file: " + path);
  }
  file.seekg(0, std::ios::end);
  size_t byte_size = file.tellg();
  file.seekg(0, std::ios::beg);
  if (byte_size != expected_elems * sizeof(uint16_t)) {
    throw std::runtime_error("Size mismatch for " + path + ": got " +
                             std::to_string(byte_size) + " bytes, expected " +
                             std::to_string(expected_elems * sizeof(uint16_t)));
  }
  std::vector<uint16_t> buf(expected_elems);
  file.read(reinterpret_cast<char *>(buf.data()), byte_size);
  return buf;
}

int main(int argc, char **argv) {
  std::string bin_dir;
  {
    std::string argv0(argv[0]);
    auto sep = argv0.rfind('/');
    bin_dir = (sep == std::string::npos) ? "." : argv0.substr(0, sep);
  }
  std::string xclbin_path = bin_dir + "/network.xclbin";
  std::string instr_path = bin_dir + "/npu_insts.bin";

  bool verbose = false;
  unsigned int warmup_iters = 5;
  unsigned int meas_iters = 100;
  if (const char *env = std::getenv("DEEPLOY_WARMUP"))
    warmup_iters = static_cast<unsigned int>(std::strtoul(env, nullptr, 10));
  if (const char *env = std::getenv("DEEPLOY_ITERATIONS"))
    meas_iters = static_cast<unsigned int>(std::strtoul(env, nullptr, 10));

  std::vector<std::string> positional;
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "-v" || arg == "--verbose" || arg == "-vv") {
      verbose = true;
    } else if ((arg == "--warmup" || arg == "-w") && i + 1 < argc) {
      warmup_iters = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
    } else if ((arg == "--iterations" || arg == "-n") && i + 1 < argc) {
      meas_iters = static_cast<unsigned int>(std::strtoul(argv[++i], nullptr, 10));
    } else if (!arg.empty() && arg[0] != '-') {
      positional.push_back(arg);
    }
  }
  if (positional.size() >= 1)
    xclbin_path = positional[0];
  if (positional.size() >= 2)
    instr_path = positional[1];

  if (meas_iters < 1)
    meas_iters = 1;

  // 1. Open device, register xclbin, build kernel
  auto device = xrt::device(0);
  auto xclbin = xrt::xclbin(xclbin_path);
  device.register_xclbin(xclbin);
  xrt::hw_context context(device, xclbin.get_uuid());
  auto kernel = xrt::kernel(context, "MLIR_AIE");

  // 2. Read NPU instruction binary
  std::vector<uint32_t> instr_v = read_instr_binary(instr_path);
  size_t n_instr = instr_v.size();

  constexpr size_t elem_size = sizeof(uint16_t); // BF16 = 2 bytes

  // 2a. In file mode, load .bin sidecars next to the binary. In embed mode,
  // these stay empty and the embedded `testInputVector` / `testOutputVector`
  // arrays are used instead.
  std::vector<std::vector<uint16_t>> file_inputs;
  std::vector<std::vector<uint16_t>> file_outputs;
#ifdef USE_FILE_INPUTS
  file_inputs.reserve(N_INPUTS);
  for (unsigned int i = 0; i < N_INPUTS; ++i) {
    file_inputs.push_back(
        read_bf16_binary(bin_dir + "/" + kInputFilenames[i], kInputElems[i]));
  }
#endif
#ifdef USE_FILE_OUTPUTS
  file_outputs.reserve(N_OUTPUTS);
  for (unsigned int i = 0; i < N_OUTPUTS; ++i) {
    file_outputs.push_back(
        read_bf16_binary(bin_dir + "/" + kOutputFilenames[i], kOutputElems[i]));
  }
#endif

  auto get_input_data = [&](unsigned int i) -> const uint16_t * {
#ifdef USE_FILE_INPUTS
    return file_inputs[i].data();
#else
    return static_cast<const uint16_t *>(testInputVector[i]);
#endif
  };
  auto get_output_data = [&](unsigned int i) -> const uint16_t * {
#ifdef USE_FILE_OUTPUTS
    return file_outputs[i].data();
#else
    return static_cast<const uint16_t *>(testOutputVector[i]);
#endif
  };

  // 3. Allocate XRT buffers.
  //    Kernel arg layout (matches aiecc.py output):
  //      0: opcode  1: bo_instr  2: instr_len
  //      3..3+N_INPUTS-1:                           inputs
  //      3+N_INPUTS..3+N_INPUTS+N_OUTPUTS-1:        outputs
  //      3+N_INPUTS+N_OUTPUTS:                      ctrlpkts
  //      3+N_INPUTS+N_OUTPUTS+1:                    trace
  constexpr unsigned int n_data_args = N_INPUTS + N_OUTPUTS;
  constexpr unsigned int ctrlpkts_gid = 3u + n_data_args;
  constexpr unsigned int trace_gid = ctrlpkts_gid + 1u;

  auto bo_instr = xrt::bo(device, n_instr * sizeof(uint32_t),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));

  std::vector<xrt::bo> bo_inputs;
  bo_inputs.reserve(N_INPUTS);
  for (unsigned int i = 0; i < N_INPUTS; ++i) {
    bo_inputs.emplace_back(device, kInputElems[i] * elem_size,
                           XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3u + i));
  }

  std::vector<xrt::bo> bo_outputs;
  bo_outputs.reserve(N_OUTPUTS);
  for (unsigned int i = 0; i < N_OUTPUTS; ++i) {
    bo_outputs.emplace_back(device, kOutputElems[i] * elem_size,
                            XRT_BO_FLAGS_HOST_ONLY,
                            kernel.group_id(3u + N_INPUTS + i));
  }

  auto bo_ctrlpkts =
      xrt::bo(device, 8, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(ctrlpkts_gid));

  constexpr size_t trace_alloc =
      TRACE_BUFFER_SIZE > 0 ? TRACE_BUFFER_SIZE * 4 : 1;
  auto bo_trace = xrt::bo(device, trace_alloc, XRT_BO_FLAGS_HOST_ONLY,
                          kernel.group_id(trace_gid));

  if constexpr (TRACE_BUFFER_SIZE > 0) {
    std::memset(bo_trace.map<void *>(), 0, trace_alloc);
    bo_trace.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // 4. Copy host data into device buffers
  std::memcpy(bo_instr.map<uint32_t *>(), instr_v.data(),
              n_instr * sizeof(uint32_t));
  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  for (unsigned int i = 0; i < N_INPUTS; ++i) {
    std::memcpy(bo_inputs[i].map<void *>(), get_input_data(i),
                kInputElems[i] * elem_size);
    bo_inputs[i].sync(XCL_BO_SYNC_BO_TO_DEVICE);
  }

  // 5. Launch kernel using set_arg (handles arbitrary arity)
  auto run = xrt::run(kernel);
  unsigned int arg_idx = 0;
  unsigned int opcode = 3;
  run.set_arg(arg_idx++, opcode);
  run.set_arg(arg_idx++, bo_instr);
  run.set_arg(arg_idx++, static_cast<uint32_t>(n_instr));
  for (auto &bo : bo_inputs)
    run.set_arg(arg_idx++, bo);
  for (auto &bo : bo_outputs)
    run.set_arg(arg_idx++, bo);
  run.set_arg(arg_idx++, bo_ctrlpkts);
  run.set_arg(arg_idx++, bo_trace);

  // 5a. Warmup iterations.
  for (unsigned int i = 0; i < warmup_iters; ++i) {
    run.start();
    run.wait();
  }

  // 5b. Measured iterations.
  using clk = std::chrono::steady_clock;
  std::vector<double> latencies_us;
  latencies_us.reserve(meas_iters);
  for (unsigned int i = 0; i < meas_iters; ++i) {
    auto t0 = clk::now();
    run.start();
    run.wait();
    auto t1 = clk::now();
    latencies_us.push_back(
        std::chrono::duration<double, std::micro>(t1 - t0).count());
  }

  // 6. Sync outputs back and compare against golden
  for (auto &bo : bo_outputs)
    bo.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  // 6a. Latency statistics. Throughput is computed from the median latency over the total bytes moved across ShimDMAs (one round trip).
  std::sort(latencies_us.begin(), latencies_us.end());
  const double lat_min = latencies_us.front();
  const double lat_max = latencies_us.back();
  const double lat_med = (meas_iters % 2 == 1)
      ? latencies_us[meas_iters / 2]
      : 0.5 * (latencies_us[meas_iters / 2 - 1] + latencies_us[meas_iters / 2]);
  double lat_sum = 0.0;
  for (double v : latencies_us)
    lat_sum += v;
  const double lat_mean = lat_sum / static_cast<double>(meas_iters);
  double lat_sq = 0.0;
  for (double v : latencies_us)
    lat_sq += (v - lat_mean) * (v - lat_mean);
  const double lat_stdev = std::sqrt(lat_sq / static_cast<double>(meas_iters));

  size_t total_in_bytes = 0;
  for (unsigned int i = 0; i < N_INPUTS; ++i)
    total_in_bytes += kInputElems[i] * elem_size;
  size_t total_out_bytes = 0;
  for (unsigned int i = 0; i < N_OUTPUTS; ++i)
    total_out_bytes += kOutputElems[i] * elem_size;
  const size_t total_bytes = total_in_bytes + total_out_bytes;
  const double throughput_gbps =
      (lat_med > 0.0)
          ? (static_cast<double>(total_bytes) / (lat_med * 1e-6)) / 1e9
          : 0.0;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "Performance:\n";
  std::cout << "  warmup iters  : " << warmup_iters << "\n";
  std::cout << "  measured iters: " << meas_iters << "\n";
  std::cout << "  latency [us]  : "
            << "min=" << lat_min << "  median=" << lat_med
            << "  mean=" << lat_mean << "  max=" << lat_max
            << "  stdev=" << lat_stdev << "\n";
  std::cout << "  element-wise throughput estimate  : " << throughput_gbps << " GB/s ("
            << total_bytes << " bytes / median latency)\n";

  int errors = 0;
  size_t total_elems = 0;
  for (unsigned int o = 0; o < N_OUTPUTS; ++o) {
    const uint16_t *hw_out = bo_outputs[o].map<const uint16_t *>();
    const uint16_t *golden_out = get_output_data(o);
    const size_t n = kOutputElems[o];
    total_elems += n;
    for (size_t i = 0; i < n; ++i) {
      bool match =
          bf16_nearly_equal(hw_out[i], golden_out[i], BF16_TOLERANCE_ULPS);
      if (!match) {
        ++errors;
        if (errors <= 10) {
          std::cerr << "  Mismatch out[" << o << "][" << i
                    << "]: hw=" << bf16_to_float(hw_out[i]) << " (0x"
                    << std::hex << hw_out[i] << std::dec << ")"
                    << "  ref=" << bf16_to_float(golden_out[i]) << " (0x"
                    << std::hex << golden_out[i] << std::dec << ")"
                    << "  diff="
                    << std::fabs(bf16_to_float(hw_out[i]) -
                                 bf16_to_float(golden_out[i]))
                    << "\n";
        }
      }
      if (verbose) {
        float hw_f = bf16_to_float(hw_out[i]);
        float ref_f = bf16_to_float(golden_out[i]);
        std::cout << "out[" << o << "][" << i << "] hw=" << hw_f
                  << "  ref=" << ref_f << "  diff=" << std::fabs(hw_f - ref_f)
                  << (match ? "" : "  *** MISMATCH") << "\n";
      }
    }
  }

  // Output format required by testUtils/core/output_parser.py
  std::cout << "Errors: " << errors << " out of " << total_elems << "\n";

  // 7. Read back trace data and write to trace.txt
  if constexpr (TRACE_BUFFER_SIZE > 0) {
    bo_trace.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

    const uint32_t *trace_data = bo_trace.map<const uint32_t *>();
    size_t trace_words = TRACE_BUFFER_SIZE / sizeof(uint32_t);

    std::string trace_path = bin_dir + "/trace.txt";
    std::ofstream trace_file(trace_path);
    if (trace_file.is_open()) {
      for (size_t i = 0; i < trace_words; ++i) {
        if (trace_data[i] != 0) {
          trace_file << std::hex << std::setfill('0') << std::setw(8)
                     << trace_data[i] << "\n";
        }
      }
      trace_file.close();
      std::cout << "Trace written to " << trace_path << "\n";
    } else {
      std::cerr << "Warning: could not open " << trace_path << " for writing\n";
    }
  }

  return (errors == 0) ? 0 : 1;
}
