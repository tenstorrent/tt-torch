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
import csv
import json


class Perf:
    @staticmethod
    def get_ttmetal_home_path():
        return os.environ.get(
            "TT_METAL_HOME",
            "third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal",
        )

    def __init__(self):
        self.tracy_capture_tool_path = f"{self.get_ttmetal_home_path()}/../tt-metal-build/tools/profiler/bin/capture-release"  # tt-metal-build/tools/profiler/bin/capture-release
        self.tracy_csvexport_tool_path = f"{self.get_ttmetal_home_path()}/../tt-metal-build/tools/profiler/bin/csvexport-release"

        self.profiler_logs_dir = (
            f"{self.get_ttmetal_home_path()}/generated/profiler/.logs"
        )
        self.tracy_file_path = "tracy_profile_log_host.tracy"
        self.tracy_ops_times_file_path = "tracy_ops_times.csv"
        self.tracy_ops_data_file_path = "tracy_ops_data.csv"
        self.profiler_device_ops_perf_trace_file_path = (
            "results/perf/device_ops_perf_trace.csv"
        )

        self.profiler_device_side_log_path = (
            f"install/tt-metal/generated/profiler/.logs/profile_log_device.csv"
        )
        self.expected_profiler_device_side_log_path = f"{self.get_ttmetal_home_path()}/generated/profiler/.logs/profile_log_device.csv"
        self.profiler_csv_file_path = f"{self.get_ttmetal_home_path()}/generated/profiler/reports/ops_perf_results.csv"

        FileManager.remove_directory(self.profiler_logs_dir)
        FileManager.create_directory(self.profiler_logs_dir)

        FileManager.remove_file(self.tracy_ops_times_file_path)
        FileManager.remove_file(self.tracy_ops_data_file_path)

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

        # port = 8086

        print(f"selected port={port}")

        os.environ["TT_METAL_DEVICE_PROFILER"] = "1"
        os.environ["TT_METAL_CLEAR_L1"] = "1"
        os.environ["TT_METAL_DEVICE_PROFILER_DISPATCH"] = "0"

        tracy_capture_tool_command = (
            f"{self.tracy_capture_tool_path} -o {self.tracy_file_path} -f -p {port}"
        )
        self.tracy_capture_tool_process = subprocess.Popen(
            tracy_capture_tool_command, shell=True  # ,env=os.environ.copy()
        )

        # self.tracy_capture_tool_process.terminate() # doesn't work when shell=True
        # os.killpg(os.getpgid(self.tracy_capture_tool_process.pid), signal.SIGTERM)

    def close_capture_tool_process(self):
        if self.tracy_capture_tool_process is None:
            return

        print("trying to close capture tool process")

        try:
            # block until tracy capture tool exits with T/O
            # self.tracy_capture_tool_process.terminate()
            self.tracy_capture_tool_process.communicate(timeout=15)
        except subprocess.TimeoutExpired as e:
            self.tracy_capture_tool_process.terminate()
            self.tracy_capture_tool_process.communicate()
            raise Exception(
                f"No profiling data could be captured. Please make sure you are on the correct build"
            )

        print("done closing capture tool process")

    def process_csvexport(self):
        print("starting process csv export")

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
        print("done writing tracy ops times ")
        with open(self.tracy_ops_data_file_path, "w") as csv_file:
            subprocess.run(
                f'{self.tracy_csvexport_tool_path} -m -s ";" {self.tracy_file_path}',
                shell=True,
                check=True,
                stdout=csv_file,
                stderr=subprocess.DEVNULL,
            )
        print("done writing tracy ops data ")

    def copy_files_to_tt_metal(self):
        # ref:
        # copy all relevant files to correct folder directory (metal hardcoded path, need to make more dynamic from metal library)
        print("copying artifacts ")

        FileManager.copy_file(self.profiler_logs_dir, self.tracy_file_path)
        FileManager.copy_file(self.profiler_logs_dir, self.tracy_ops_times_file_path)
        FileManager.copy_file(self.profiler_logs_dir, self.tracy_ops_data_file_path)

        FileManager.copy_file(
            self.expected_profiler_device_side_log_path,
            self.profiler_device_side_log_path,
        )
        FileManager.copy_file(self.profiler_logs_dir, self.tracy_ops_data_file_path)

    def run_ttmetal_process_ops(self):
        # need to temporary add these sys paths so TTRT whls can find the `process_ops` function
        # ideally we want process_ops to be in a standalone module we can import from tt_metal
        # ref: https://github.com/tenstorrent/tt-metal/blob/5b24e448480c86c6e74563745b58f8c95fa81237/tt_metal/tools/profiler/process_ops_logs.py#L37
        sys.path.append(f"{self.get_ttmetal_home_path()}")
        sys.path.append(f"{self.get_ttmetal_home_path()}/ttnn")
        from tt_metal.tools.profiler.process_ops_logs import process_ops

        process_ops(
            None, None, False, False  # generate date & device only
        )  # where does this expect the files to be?

    def post_process_ops(self):
        # Add post-processing steps to insert location data into the ops_perf data file
        with open(self.profiler_csv_file_path, "r") as perf_file:
            perf_reader = csv.DictReader(perf_file)
            headers = list(perf_reader.fieldnames) + ["LOC"]
            perf_data = list(perf_reader)

        with open(self.profiler_csv_file_path, "w+") as perf_file, open(
            self.tracy_ops_data_file_path, "r"
        ) as message_file:
            message_reader = csv.reader(message_file, delimiter=";")
            ops_index = 0
            prev = None
            for message in message_reader:
                message = message[0]  # Don't need timestamp information
                if message.startswith("`"):
                    # This is a TTNN Message
                    # The location data is now in the previous message
                    # The order of data is maintained in perf_data so as the messages are received, they update the id last encountered.
                    # Now that we have a new message, we can update the location data from the previous message
                    if prev:
                        # Get the location data from the previous message and add it as new data for the perf_data (as a new col)
                        if len(perf_data) > ops_index:
                            perf_data[ops_index]["LOC"] = prev
                            ops_index += 1
                else:
                    prev = message
            perf_writer = csv.DictWriter(perf_file, fieldnames=headers)
            perf_writer.writeheader()
            for row in perf_data:
                perf_writer.writerow(row)

        FileManager.create_file(self.profiler_device_ops_perf_trace_file_path)

        # Trim out non-loc associated ops (And other post processing)
        with open(self.profiler_csv_file_path, "r") as perf_file, open(
            self.profiler_device_ops_perf_trace_file_path, "w+"
        ) as report_file:
            perf_reader = csv.DictReader(perf_file)
            headers = perf_reader.fieldnames
            headers = [headers[-1]] + headers[:-1]  # move LOC to first column

            ops_writer = csv.DictWriter(report_file, headers)
            ops_writer.writeheader()
            for row in perf_reader:
                if "loc" in row["LOC"]:
                    ops_writer.writerow(row)

    def cleanup(self):
        FileManager.remove_file(self.tracy_ops_times_file_path)
        FileManager.remove_file(self.tracy_ops_data_file_path)
        FileManager.remove_file(self.tracy_file_path)
