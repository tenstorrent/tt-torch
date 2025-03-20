# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import glob
import os
from xml.etree.ElementTree import ElementTree, Element, parse
from tt_torch.tools.crashsafe_utils import rewrite_crashsafe_xml


def merge_junit_reports(input_files, output_file):
    """
    Merge multiple JUnit XML files into a single JUnit XML file.

    Args:
        input_files (list): List of paths to JUnit XML files to merge.
        output_file (str): Path to the output merged JUnit XML file.
    """
    if not input_files:
        print("No input files found to merge.")
        return

    # Parse the first file as the base
    base_tree = parse(input_files[0])
    base_root = base_tree.getroot()

    # Find the single <testsuite> element in the base file
    base_testsuite = base_root.find(".//testsuite")
    if base_testsuite is None:
        print(f"Error: No <testsuite> element found in {input_files[0]}")
        return

    # Merge the rest of the files into the base <testsuite>
    for file in input_files[1:]:
        tree = parse(file)
        root = tree.getroot()

        # Find the <testsuite> element in the current file
        testsuite = root.find(".//testsuite")
        if testsuite is None:
            print(f"Warning: No <testsuite> element found in {file}, skipping.")
            continue

        # Append all <testcase> elements from the current <testsuite> to the base <testsuite>
        for testcase in testsuite.findall(".//testcase"):
            base_testsuite.append(testcase)

    # Update the <testsuite> test count to be internally consistent
    base_testsuite.set("tests", str(len(input_files)))  # Update the number of tests

    # Write the merged XML to the output file
    base_tree.write(output_file, encoding="utf-8", xml_declaration=True)
    print(f"Merged {len(input_files)} files into {output_file}")


def process_and_merge_reports(search_pattern, output_file):
    """
    Process and merge JUnit XML reports.

    Args:
        search_pattern (str): Glob pattern to match JUnit XML files.
        output_file (str): Path to the output merged JUnit XML file.
    """
    # Find all files matching the search pattern
    input_files = glob.glob(search_pattern)
    if not input_files:
        print(f"No files found matching pattern: {search_pattern}")
        return

    print(f"Found {len(input_files)} files to process.")

    # Process each file with rewrite_crashsafe_xml
    for file in input_files:
        print(f"Processing file: {file}")
        rewrite_crashsafe_xml(file)

    # Merge all processed files into a single output file
    merge_junit_reports(input_files, output_file)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            'Usage: python tt_torch/tools/postprocess_test_reports.py "<search_pattern>" "<output_file>"'
        )
        sys.exit(1)

    search_pattern = sys.argv[1]
    output_file = sys.argv[2]

    process_and_merge_reports(search_pattern, output_file)
