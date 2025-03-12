# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# from tt_mlir import is_runtime_debug_enabled  # eqv for perf enabled
import os
import typing
import subprocess
import time
import socket
import signal
import sys
import shutil
from utils import FileManager


class Perf:
    @staticmethod
    def get_ttmetal_home_path():
        return os.environ.get(
            "TT_METAL_HOME",
            "third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal",
        )  # /localdev/jameszianxu/tracy/tt-torch/third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal

    def __init__(self):
        self.tracy_capture_tool_path = f"{self.get_ttmetal_home_path()}/../tt-metal-build/tools/profiler/bin/capture-release"  # tt-metal-build/tools/profiler/bin/capture-release
        self.tracy_csvexport_tool_path = f"{self.get_ttmetal_home_path()}/../tt-metal-build/tools/profiler/bin/csvexport-release"

        self.profiler_logs_dir = (
            f"{self.get_ttmetal_home_path()}/generated/profiler/.logs"
        )
        self.tracy_file_path = "tracy_profile_log_host.tracy"
        self.tracy_ops_times_file_path = "tracy_ops_times.csv"
        self.tracy_ops_data_file_path = "tracy_ops_data.csv"
        self.profiler_device_side_log_path = f"{self.get_ttmetal_home_path()}/generated/profiler/.logs/profile_log_device.csv"
        self.profiler_csv_file_path = f"{self.get_ttmetal_home_path()}/generated/profiler/reports/ops_perf_results.csv"

        FileManager.remove_directory(self.profiler_logs_dir)
        FileManager.create_directory(self.profiler_logs_dir)

    def setup_tracy_server(self) -> None:
        def get_available_port():
            ip = socket.gethostbyname(socket.gethostname())

            for port in range(8086, 8500):
                try:
                    serv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    serv.bind((ip, port))
                    return str(port)
                except PermissionError as e:
                    pass
                except OSError as e:
                    pass
            return None

        port = get_available_port()

        if not port:
            raise Exception("No available port found")

        port = 8086

        print(f"selected port={port}")

        os.environ[
            "TT_METAL_DEVICE_PROFILER"
        ] = "1"  # carries over to other cmds run in shell

        # os.environ["TT_METAL_CLEAR_L1"] = "1"
        # os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

        tracy_capture_tool_command = (
            f"{self.tracy_capture_tool_path} -o {self.tracy_file_path} -f -p {8086}"
        )
        self.tracy_capture_tool_process = subprocess.Popen(
            tracy_capture_tool_command, shell=True  # ,env=os.environ.copy()
        )

        # self.tracy_capture_tool_process.terminate() # doesn't work when shell=True
        # os.killpg(os.getpgid(self.tracy_capture_tool_process.pid), signal.SIGTERM)

    def close_capture_tool_process(self):
        if self.tracy_capture_tool_process is None:
            return

        try:
            # block until tracy capture tool exits with T/O
            self.tracy_capture_tool_process.communicate(timeout=60)
        except subprocess.TimeoutExpired as e:
            self.tracy_capture_tool_process.terminate()
            self.tracy_capture_tool_process.communicate()
            raise Exception(
                f"No profiling data could be captured. Please make sure you are on the correct build"
            )

    def process_csvexport(self):
        with open(self.tracy_ops_times_file_path, "w") as csv_file:
            child_calls = ["CompileProgram", "HWCommandQueue_write_buffer"]
            child_calls_str = f"-x {','.join(child_calls)}"
            subprocess.run(
                f"{self.tracy_csvexport_tool_path} -u -p TT_DNN {child_calls_str} {self.tracy_file_path}",
                shell=True,
                check=True,
                stdout=csv_file,
                stderr=subprocess.DEVNULL,
            )
        print("writing tracy ops times ")
        with open(self.tracy_ops_data_file_path, "w") as csv_file:
            subprocess.run(
                f'{self.tracy_csvexport_tool_path} -m -s ";" {self.tracy_file_path}',
                shell=True,
                check=True,
                stdout=csv_file,
                stderr=subprocess.DEVNULL,
            )
        print("writing tracy ops data ")
