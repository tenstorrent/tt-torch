# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from tt_torch.tools.profile_util import Profiler
import subprocess
import os
import signal
import sys
from argparse import ArgumentParser


def tt_profile(
    test_command: str,
    output_filename: str = Profiler.DEFAULT_OUTPUT_FILENAME,
    port: int = 8086,
):
    profiler = Profiler(output_filename, port)
    profiler.assert_perf_build()
    tracy_port = profiler.setup_tracy_server()

    # Test process must NOT run in main process, (i.e. via pytest.main) or else tracy capture will deadlock with main

    # Another way to deadlock with main is if the tt_mlir bindings are imported. Something in that import / pybind path
    #   initializes a tracy client, potentially something in the initialization of tt-metal or tt-mlir that is inadvertently executed.
    #   This could even be a header import from way down in the stack.

    # The TRACY_PORT env variable is set in setup_tracy_server.

    env_vars = dict(os.environ)
    env_vars["TRACY_PORT"] = tracy_port
    env_vars["TT_METAL_DEVICE_PROFILER"] = "1"
    env_vars["TT_METAL_CLEAR_L1"] = "1"
    env_vars["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"
    # env_vars["TT_METAL_PROFILER_SYNC"] = "1"  # fix desync between host and device calls - disabled due to CI issue

    testProcess = subprocess.Popen(
        [test_command], shell=True, env=env_vars, preexec_fn=os.setsid
    )

    def signal_handler(sig, frame):
        os.killpg(os.getpgid(testProcess.pid), signal.SIGTERM)
        profiler.tracy_capture_tool_process.terminate()
        profiler.tracy_capture_tool_process.communicate()
        sys.exit(3)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    testProcess.communicate()  # block until the test process exits

    profiler.close_capture_tool_process()
    profiler.process_csvexport()
    profiler.copy_files_to_tt_metal()
    profiler.run_ttmetal_process_ops()
    profiler.post_process_ops()
    profiler.cleanup()


def main():
    """
    This script is used to wrap pytests and generate Tracy profiling data for the tests,
    gathering device-side performance data associated with individual torchfx ops.

    Usage:
        python tt_profile.py test_command -o output_name

    Examples:
        python tt_torch/tools/tt_profile.py "pytest -svv tests/models/mnist/test_mnist.py::test_mnist_train[full-eval-single_device]"

    Notes:
        Providing an output name is optional and defaults to 'device_ops_perf_trace.csv'."""

    parser = ArgumentParser()
    parser.add_argument("test_command", type=str, help="The test command to run")
    parser.add_argument(
        "-o",
        "--output_path",
        help="Output file path",
        type=str,
        default=Profiler.DEFAULT_OUTPUT_FILENAME,
    )
    parser.add_argument(
        "-p",
        "--port",
        help="Output file path",
        type=int,
        default=8086,  # tracy client default port
    )

    args = parser.parse_args()

    tt_profile(args.test_command, args.output_path, args.port)


if __name__ == "__main__":
    main()
