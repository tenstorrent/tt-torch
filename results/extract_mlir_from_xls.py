# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to extract MLIR dialect snippets from an XLS file and run compiler commands on them.

Usage:
  python extract_mlir_from_xls.py --file path/to/file.xls --status 7 --error "Some Error message"
"""

import argparse
import os
import re
import subprocess
import pandas as pd
from pathlib import Path
import datetime
import time


def sanitize_filename(text):
    """Convert text to a valid filename by replacing invalid characters with underscores."""
    if not text:
        return "unknown"
    # Replace invalid filename characters and spaces with underscores
    return re.sub(r"[^\w\-\.]", "_", str(text).lower())


def find_latest_ir_column(row, ir_columns):
    """Find the rightmost non-empty IR column in the row."""
    for column in reversed(list(ir_columns.keys())):
        ir_content = row.get(column, "")
        if pd.notna(ir_content) and ir_content:
            return column, ir_columns[column]
    return None, None


def process_xls(file_path, status_filter, error_filter, dump_all_ir):
    """
    Process XLS file to extract MLIR dialect snippets based on filters.

    Args:
        file_path: Path to the XLS file
        status_filter: Integer status to filter rows by
        error_filter: String pattern to match in the Compile Error column
        dump_all_ir: Whether to dump all IRs or just the latest in the pipeline
    """
    print(f"Processing {file_path}")
    print(f"Filtering for Status={status_filter} and Error containing '{error_filter}'")
    print(f"Dumping {'all IRs' if dump_all_ir else 'only the latest IR'}")

    # Read the specific sheet from the Excel file
    try:
        df = pd.read_excel(file_path, sheet_name="All Ops (Not Executing)")
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return

    # Create output directory
    output_dir = Path("mlir_files")
    output_dir.mkdir(exist_ok=True)

    # Log file for compiler outputs
    log_file = output_dir / "compiler_output.log"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_file, "w") as logf:
        logf.write(f"Processing results for file: {os.path.abspath(file_path)}\n")
        logf.write(f"Timestamp: {timestamp}\n")
        logf.write(
            f"Parameters: status={status_filter}, error='{error_filter}', dump_all_ir={dump_all_ir}\n"
        )
        logf.write("=" * 80 + "\n\n")

    # Define the IR columns to extract
    ir_columns = {
        "Torch IR": "torch_ir",
        "Raw SHLO": "raw_shlo",
        "Raw TTIR": "raw_ttir",
        "Raw TTNNIR": "raw_ttnnir",
    }

    # Counter for matching rows
    match_count = 0

    # Dictionary to track used base filenames and their counts
    used_filenames = {}

    # Process each row in the dataframe
    for idx, row in df.iterrows():
        # Check if the row matches our filters
        if row["Status"] != status_filter:
            continue

        error_cell = str(row.get("Compile Error", ""))
        if error_filter and error_filter.lower() not in error_cell.lower():
            continue

        match_count += 1

        # Create a base name for this match
        torch_name = sanitize_filename(row.get("Torch Name", f"op_{idx}"))
        models = sanitize_filename(row.get("Models", "unknown_model"))
        status = sanitize_filename(row.get("Status", "unknown_status"))

        base_name = f"{torch_name}_{models}_status_{status}"

        # Handle filename collisions
        if base_name in used_filenames:
            used_filenames[base_name] += 1
            unique_base_name = f"{base_name}_{used_filenames[base_name]}"
        else:
            used_filenames[base_name] = 1
            unique_base_name = base_name

        print(f"Processing match #{match_count}: {unique_base_name}")

        # Determine which IR(s) to extract and save
        if dump_all_ir:
            # Extract and save all IR snippets
            columns_to_process = list(ir_columns.items())
        else:
            # Find the latest (rightmost) non-empty IR column
            latest_column, latest_suffix = find_latest_ir_column(row, ir_columns)
            if latest_column:
                columns_to_process = [(latest_column, latest_suffix)]
            else:
                columns_to_process = []
                print(f"  No IR content found for {unique_base_name}")

        # Process each selected IR column
        for column, suffix in columns_to_process:
            ir_content = row.get(column, "")
            if pd.notna(ir_content) and ir_content:
                file_name = f"{unique_base_name}_{suffix}.mlir"
                ir_file_path = output_dir / file_name

                with open(ir_file_path, "w") as f:
                    f.write(str(ir_content))

                print(f"  Wrote {file_name}")

                # Run ttmlir-opt on the file
                try:
                    # Select the appropriate command based on IR type
                    if column == "Raw SHLO":
                        cmd = [
                            "ttmlir-opt",
                            "--stablehlo-to-ttir-pipeline=enable-arith-to-stablehlo=true enable-composite-to-call=true",
                            str(ir_file_path),
                        ]
                    elif column == "Raw TTIR":
                        cmd = [
                            "ttmlir-opt",
                            "--ttir-to-ttnn-backend-pipeline",
                            str(ir_file_path),
                        ]
                    else:
                        cmd = ["ttmlir-opt", str(ir_file_path)]

                    print(f"  Running: {' '.join(cmd)}")

                    # Measure execution time
                    start_time = time.time()
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=45,  # Add timeout to prevent hanging
                    )
                    end_time = time.time()
                    execution_time = end_time - start_time

                    print(f"  Command completed in {execution_time:.2f} seconds")

                    # Log the results
                    with open(log_file, "a") as logf:
                        logf.write(f"File: {file_name}\n")
                        logf.write(f"IR Type: {column}\n")
                        logf.write(f"Torch Name: {row.get('Torch Name', '')}\n")
                        logf.write(f"Models: {row.get('Models', '')}\n")
                        logf.write(f"Command: {' '.join(cmd)}\n")
                        logf.write(f"Return code: {result.returncode}\n")
                        logf.write(f"Execution time: {execution_time:.2f} seconds\n")

                        if result.stdout:
                            logf.write("STDOUT:\n")
                            logf.write(result.stdout)
                            logf.write("\n")

                        if result.stderr:
                            logf.write("STDERR:\n")
                            logf.write(result.stderr)
                            logf.write("\n")

                        logf.write("-" * 80 + "\n\n")

                except Exception as e:
                    print(f"  Error running ttmlir-opt: {e}")
                    with open(log_file, "a") as logf:
                        logf.write(f"File: {file_name}\n")
                        logf.write(f"IR Type: {column}\n")
                        logf.write(f"Error running ttmlir-opt: {e}\n")
                        logf.write("-" * 80 + "\n\n")

    print(f"Processed {match_count} matching rows")
    print(f"Results saved to {output_dir}")
    print(f"Compiler output logged to {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Process XLS file to extract MLIR dialect snippets"
    )
    parser.add_argument("--file", required=True, help="Path to the XLS file to process")
    parser.add_argument(
        "--status", type=int, required=True, help="Status value to filter by"
    )
    parser.add_argument(
        "--error", default="", help="Error message to filter by (substring match)"
    )
    parser.add_argument(
        "--all-ir",
        action="store_true",
        help="Dump all IR types instead of just the latest in the pipeline",
    )

    args = parser.parse_args()

    # Make sure the file exists
    if not os.path.exists(args.file):
        print(f"Error: File {args.file} does not exist")
        return 1

    process_xls(args.file, args.status, args.error, args.all_ir)
    return 0


if __name__ == "__main__":
    exit(main())
