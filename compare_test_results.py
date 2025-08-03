#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Script to compare test results between two XML files.
Extracts model names and test case names from the old XML and checks
if the model names exist in the new XML file.
"""

import sys
import xml.etree.ElementTree as ET
import re
from typing import Set, List, Tuple


def extract_model_names_from_xml(xml_file_path: str) -> Set[str]:
    """
    Extract all unique model names from an XML file.

    Args:
        xml_file_path: Path to the XML file

    Returns:
        Set of unique model names found in the XML file
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        model_names = set()

        # Find all testcase elements
        for testcase in root.findall(".//testcase"):
            # Look for properties within each testcase
            properties = testcase.find("properties")
            if properties is not None:
                # Find the model_name property
                for prop in properties.findall("property"):
                    if prop.get("name") == "model_name":
                        model_name = prop.get("value")
                        if model_name:
                            model_names.add(model_name)
                        break

        return model_names

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {xml_file_path}", file=sys.stderr)
        sys.exit(1)


def extract_testcases_from_old_xml(xml_file_path: str) -> List[Tuple[str, str, str]]:
    """
    Extract model names, test case names, and parallelism from the old XML file.

    Args:
        xml_file_path: Path to the old XML file

    Returns:
        List of tuples containing (model_name, testcase_name, parallelism)
    """
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        testcases_info = []

        # Find all testcase elements
        for testcase in root.findall(".//testcase"):
            testcase_name = testcase.get("name", "Unknown")

            # Look for properties within each testcase
            properties = testcase.find("properties")
            model_name = None
            parallelism = "Unknown"

            if properties is not None:
                # Find the model_name and tags properties
                for prop in properties.findall("property"):
                    prop_name = prop.get("name")
                    prop_value = prop.get("value")

                    if prop_name == "model_name":
                        model_name = prop_value
                    elif prop_name == "tags" and prop_value:
                        # Extract parallelism from tags property
                        # Tags property contains a string representation of a dict
                        parallelism_match = re.search(
                            r"'parallelism':\s*'([^']+)'", prop_value
                        )
                        if parallelism_match:
                            parallelism = parallelism_match.group(1)

            # Only add if we found a model name
            if model_name:
                testcases_info.append((model_name, testcase_name, parallelism))

        return testcases_info

    except ET.ParseError as e:
        print(f"Error parsing XML file {xml_file_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except FileNotFoundError:
        print(f"File not found: {xml_file_path}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to compare test results between two XML files."""
    if len(sys.argv) != 3:
        print(
            "Usage: python compare_test_results.py <old_xml> <new_xml>", file=sys.stderr
        )
        sys.exit(1)

    old_xml_path = sys.argv[1]
    new_xml_path = sys.argv[2]

    print(f"Comparing test results:")
    print(f"Old XML: {old_xml_path}")
    print(f"New XML: {new_xml_path}")
    print()

    # Extract model names from the new XML file
    new_model_names = extract_model_names_from_xml(new_xml_path)

    # Extract testcase information from the old XML file
    old_testcases = extract_testcases_from_old_xml(old_xml_path)

    if not old_testcases:
        print("No testcases with model names found in the old XML file.")
        return

    print(f"Found {len(old_testcases)} testcases in old XML")
    print(f"Found {len(new_model_names)} unique model names in new XML")
    print()

    # Print header
    print(
        f"{'In New XML':<13} {'Parallelism':<22} {'Model Name':<90} {'Test Case Name':<50}"
    )
    print("-" * 180)

    # Compare and print results
    found_count = 0
    for model_name, testcase_name, parallelism in old_testcases:
        found_marker = "YES" if model_name in new_model_names else "NO"
        if model_name in new_model_names:
            found_count += 1

        print(
            f"{found_marker:<13} {parallelism:<22} {model_name:<90} {testcase_name:<50}"
        )

    print()
    print(f"Summary: {found_count}/{len(old_testcases)} model names found in new XML")


if __name__ == "__main__":
    main()
