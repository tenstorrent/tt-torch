# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import subprocess
import time
import socket
import sys
import csv
from tt_torch.tools.filemanager import FileManager


class Profiler:
    DEFAULT_OUTPUT_FILENAME = "device_ops_perf_trace.csv"
    port = 8086

    @staticmethod
    def get_ttmetal_home_path():
        return os.environ.get(
            "TT_METAL_HOME",
            "third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal",
        )

    def __init__(
        self, output_filename: str = DEFAULT_OUTPUT_FILENAME, port: int = 8086
    ):
        self.tracy_capture_tool_path = (
            f"{self.get_ttmetal_home_path()}/tools/profiler/bin/capture-release"
        )
        self.tracy_csvexport_tool_path = (
            f"{self.get_ttmetal_home_path()}/tools/profiler/bin/csvexport-release"
        )

        self.check_install_tt_metal_tool_binaries()

        self.profiler_logs_dir = (
            f"{self.get_ttmetal_home_path()}/generated/profiler/.logs"
        )
        self.tracy_file_path = "tracy_profile_log_host.tracy"
        self.tracy_ops_times_file_path = "tracy_ops_times.csv"
        self.tracy_ops_data_file_path = "tracy_ops_data.csv"
        self.profiler_device_side_log_path = (
            f"install/tt-metal/generated/profiler/.logs/profile_log_device.csv"
        )
        self.expected_profiler_device_side_log_path = f"{self.get_ttmetal_home_path()}/generated/profiler/.logs/profile_log_device.csv"
        self.profiler_report_csv_path = f"{self.get_ttmetal_home_path()}/generated/profiler/reports/ops_perf_results.csv"
        self.profile_ops_perf_report = f"results/perf/{output_filename}"

        FileManager.remove_directory(self.profiler_logs_dir)
        FileManager.create_directory(self.profiler_logs_dir)

        FileManager.remove_file(self.tracy_ops_times_file_path)
        FileManager.remove_file(self.tracy_ops_data_file_path)

        self.port = port

    def check_install_tt_metal_tool_binaries(self):
        this_dir = os.path.dirname(__file__)
        metal_bin_dir = os.path.join(
            this_dir,
            "..",
            "..",
            "third_party/tt-mlir/src/tt-mlir/third_party/tt-metal/src/tt-metal/build",
        )

        if FileManager.check_file_exists(
            self.tracy_capture_tool_path
        ) and FileManager.check_file_exists(self.tracy_csvexport_tool_path):
            print("Perf tool binaries were found.")
            return

        FileManager.create_directory(
            f"{self.get_ttmetal_home_path()}/tools/profiler/bin", exist_ok=True
        )

        # For a local, non-wheel case - the binaries are expected to be in the metal_bin_dir
        # and correctly placed relative to TT_METAL_HOME

        # if we can find the binary in metal_bin dir:
        if FileManager.check_file_exists(
            f"{metal_bin_dir}/tools/profiler/bin/capture-release"
        ) and FileManager.check_file_exists(
            f"{metal_bin_dir}/tools/profiler/bin/csvexport-release"
        ):
            print("Perf tool binaries not found - Installing from tree.")

            FileManager.copy_file(
                self.tracy_capture_tool_path,
                f"{metal_bin_dir}/tools/profiler/bin/capture-release",
            )
            FileManager.copy_file(
                self.tracy_csvexport_tool_path,
                f"{metal_bin_dir}/tools/profiler/bin/csvexport-release",
            )

        # For a wheel build, the binaries are in site-packages and need to be properly copied
        else:
            print("Perf tool binaries not found - Installing from wheel.")
            # expected to be @ env/venv/lib/python3.10/site-packages/tt-metal/tools/profiler/bin/[...]
            # in this context the tt_metal_home is at /env/venv/tt_metal

            site_packages_dir = os.path.join(
                self.get_ttmetal_home_path(),
                "..",
                "lib",
                "python3.10",
                "site-packages",
                "tt-metal",
                "tools",
                "profiler",
                "bin",
            )

            FileManager.copy_file(
                f"{self.get_ttmetal_home_path()}/tools/profiler/bin/capture-release",
                f"{site_packages_dir}/capture-release",
            )
            FileManager.copy_file(
                f"{self.get_ttmetal_home_path()}/tools/profiler/bin/csvexport-release",
                f"{site_packages_dir}/csvexport-release",
            )

        print(
            "Perf tool binaries were installed at ",
            self.tracy_capture_tool_path,
            self.tracy_csvexport_tool_path,
        )

    def assert_perf_build(self):
        assert FileManager.check_file_exists(
            self.tracy_capture_tool_path
        ), f"perf tool={self.tracy_capture_tool_path} does not exist - rebuild with ENABLE_PERF_TRACE"
        assert FileManager.check_file_exists(
            self.tracy_csvexport_tool_path
        ), f"perf tool={self.tracy_csvexport_tool_path} does not exist - rebuild with ENABLE_PERF_TRACE"

    def setup_tracy_server(self):
        def get_available_port():
            ip = socket.gethostbyname(socket.gethostname())

            def try_bind(port, do_raise=False):
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as serv:
                        serv.bind((ip, port))
                        return str(port)
                except (PermissionError, OSError) as e:
                    if do_raise:
                        raise e
                    return None

            # Check default port range if self.port not overridden
            if self.port == 8086:
                for port in range(8086, 8500):
                    result = try_bind(port)
                    if result:
                        return result

            # Try binding to the specified port
            else:
                return try_bind(self.port, do_raise=True)

            return None

        port = get_available_port()

        if not port:
            raise Exception("No available port found")

        tracy_capture_tool_command = (
            f"{self.tracy_capture_tool_path} -o {self.tracy_file_path} -f -p {port}"
        )

        print("Starting capture tool with command:", tracy_capture_tool_command)

        self.tracy_capture_tool_process = subprocess.Popen(
            tracy_capture_tool_command, shell=True  # ,env=os.environ.copy()
        )

        return port

    def close_capture_tool_process(self):
        if self.tracy_capture_tool_process is None:
            return
        try:
            # block until tracy capture tool exits with T/O limit.
            # this should not take long as the client should have exited and the capture tool just needs to write out the tracedump
            tracy_exit_start_time = time.time()
            self.tracy_capture_tool_process.communicate(timeout=60)
            print(
                f"Tracy capture tool has exited after {time.time() - tracy_exit_start_time} seconds."
            )
        except subprocess.TimeoutExpired as e:
            self.tracy_capture_tool_process.terminate()
            self.tracy_capture_tool_process.communicate()
            raise Exception(
                f"No profiling data could be captured. Please make sure you are on the correct build and specify a port that is not in use."
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
        with open(self.tracy_ops_data_file_path, "w") as csv_file:
            subprocess.run(
                f'{self.tracy_csvexport_tool_path} -m -s ";" {self.tracy_file_path}',
                shell=True,
                check=True,
                stdout=csv_file,
                stderr=subprocess.DEVNULL,
            )

    def copy_files_to_tt_metal(self):
        # ref comment from ttrt-perf
        # copy all relevant files to correct folder directory (metal hardcoded path, need to make more dynamic from metal library)

        FileManager.copy_file(self.profiler_logs_dir, self.tracy_file_path)
        FileManager.copy_file(self.profiler_logs_dir, self.tracy_ops_times_file_path)
        FileManager.copy_file(self.profiler_logs_dir, self.tracy_ops_data_file_path)
        FileManager.copy_file(self.profiler_logs_dir, self.tracy_ops_data_file_path)

    def run_ttmetal_process_ops(self):
        # need to temporary add these sys paths so TTRT whls can find the `process_ops` function
        # ideally we want process_ops to be in a standalone module we can import from tt_metal
        # ref: https://github.com/tenstorrent/tt-metal/blob/main/tt_metal/tools/profiler/process_ops_logs.py
        sys.path.append(f"{self.get_ttmetal_home_path()}")
        sys.path.append(f"{self.get_ttmetal_home_path()}/ttnn")
        from tt_metal.tools.profiler.process_ops_logs import process_ops

        #  Decode ops @ self.tracy_ops_data_file_path
        process_ops(None, None, False, False)

    def post_process_ops(self):
        def get_mlir_analysis_results(key):
            call_count_mapping = {}

            with open(self.tracy_ops_data_file_path, "r") as file:
                lines = iter(file)
                buffer = None

                while True:
                    # Use buffered line if available, otherwise get next
                    line = buffer if buffer else next(lines, None)
                    buffer = None

                    if line is None:
                        break  # End of file

                    # Find all the TT_DNN_DEVICE_OP under this LOC and record their global call counts
                    line = line.strip()
                    if key in line:
                        # Format of line is
                        # MLIR_OP_LOCATION;loc("/code/tt-mlir/build/test/ttmlir/Silicon/TTNN/n150/const-eval/Output/const-eval.mlir.tmp.mlir":17:14);5420869271
                        # MLIR_CONST_EVAL_OP;true;6449925338
                        parts = line.split(";")
                        data = parts[1]
                        block = []
                        for next_line in lines:
                            next_line = next_line.strip()
                            if key in next_line:
                                buffer = next_line  # Save for next outer loop
                                break
                            elif "TT_DNN_DEVICE_OP" in next_line:
                                block.append(next_line)

                        # Process the collected block. Find it's global call count and add it to the loc
                        for bline in block:
                            parts = bline.split(",")
                            # Strip and split part[3] on semicolon or space, and grab the number
                            num_part = parts[3].strip()
                            digits = ""
                            for c in num_part:
                                if c.isdigit():
                                    digits += c
                                else:
                                    break
                            global_call_count = int(digits) if digits else None
                            call_count_mapping[global_call_count] = data

            return call_count_mapping

        global_call_count_loc_mapping = get_mlir_analysis_results("MLIR_OP_LOCATION")
        global_call_count_const_eval_op_mapping = get_mlir_analysis_results(
            "MLIR_CONST_EVAL_OP"
        )
        global_call_count_program_metadata_op_mapping = get_mlir_analysis_results(
            "MLIR_PROGRAM_METADATA"
        )

        # Add location data, const_eval_op data and program metadata to profiler csv file
        FileManager.create_file(self.profile_ops_perf_report)

        with open(self.profiler_report_csv_path, mode="r", newline="") as infile, open(
            self.profile_ops_perf_report, mode="w", newline=""
        ) as outfile:
            reader = csv.DictReader(infile)
            fieldnames = reader.fieldnames + [
                "LOC",
                "CONST_EVAL_OP",
                "PROGRAM_METADATA",
            ]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                # Access the value at "GLOBAL CALL COUNT"
                local_call_count = row.get("GLOBAL CALL COUNT")
                local_call_count = int(local_call_count.strip())

                # Append the location column with its location data
                if local_call_count in global_call_count_loc_mapping.keys():
                    row["LOC"] = global_call_count_loc_mapping[local_call_count]
                else:
                    row["LOC"] = "loc(unknown)"

                # Append the const_eval_op column with its const_eval_op data
                if local_call_count in global_call_count_const_eval_op_mapping.keys():
                    row["CONST_EVAL_OP"] = global_call_count_const_eval_op_mapping[
                        local_call_count
                    ]
                else:
                    row["CONST_EVAL_OP"] = "false"

                # Append the program metadata column with its metadata
                if (
                    local_call_count
                    in global_call_count_program_metadata_op_mapping.keys()
                ):
                    row[
                        "PROGRAM_METADATA"
                    ] = global_call_count_program_metadata_op_mapping[local_call_count]
                else:
                    row["PROGRAM_METADATA"] = "{}"

                writer.writerow(row)

        print(f"Wrote report to {self.profile_ops_perf_report}")

    def cleanup(self):
        FileManager.remove_file(self.tracy_ops_times_file_path)
        FileManager.remove_file(self.tracy_ops_data_file_path)
        FileManager.remove_file(self.tracy_file_path)
