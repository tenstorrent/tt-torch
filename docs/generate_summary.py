#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
from pathlib import Path


def generate_summary(docs_dir):
    """
    Generate top level SUMMARY.md which includes references to all pages
    """
    summary_lines = ["# Summary"]
    summary_lines.append("- [Introduction](overview.md)")
    summary_lines.append("- [Getting Started](getting_started.md)")
    summary_lines.append("\n\n# User guide")
    summary_lines.append("- [Testing](test.md)")
    summary_lines.append("- [Controlling Compiler](controlling.md)")
    summary_lines.append("- [Pre-commit](pre_commit.md)")
    summary_lines.append("- [Profiling](profiling.md)")
    summary_lines.append("- [Adding Models](adding_models.md)")
    summary_lines.append("\n\n# Models and Operations")
    summary_lines.append("- [Supported Models](models/supported_models.md)")

    summary_lines.append("- [Operations](ops/README.md)")
    summary_lines.append("  - [StableHLO Operations](ops/stablehlo/README.md)")

    # Add all Stablehlo operations
    stablehlo_dir = Path(docs_dir) / "ops" / "stablehlo"
    if stablehlo_dir.exists():
        for file in sorted(stablehlo_dir.glob("*.md")):
            if file.name != "README.md":
                title = file.stem.replace(".", " ").title()
                rel_path = file.relative_to(docs_dir)
                summary_lines.append(f"    - [{title}]({str(rel_path)})")

    summary_lines.append("  - [TTNN Operations](ops/ttnn/README.md)")

    # Add all TTNN operations
    ttnn_dir = Path(docs_dir) / "ops" / "ttnn"
    if ttnn_dir.exists():
        for file in sorted(ttnn_dir.glob("*.md")):
            if file.name != "README.md":
                title = file.stem.replace(".", " ").title()
                rel_path = file.relative_to(docs_dir)
                summary_lines.append(f"    - [{title}]({str(rel_path)})")

    return "\n".join(summary_lines)


def ensure_readme_files(docs_dir):
    """
    Ensure README.md files exist in necessary directories
    """
    required_readmes = [
        "ops/README.md",
        "ops/stablehlo/README.md",
        "ops/ttnn/README.md",
    ]

    for readme_path in required_readmes:
        full_path = Path(docs_dir) / readme_path
        if not full_path.exists():
            full_path.parent.mkdir(parents=True, exist_ok=True)
            with open(full_path, "w") as f:
                section_name = full_path.parent.name.title()
                f.write(f"# {section_name} Documentation\n\n")
                f.write(
                    f"This section contains documentation for {section_name} operations.\n"
                )


def main():
    docs_src_dir = Path("./docs/src")

    if not docs_src_dir.exists():
        print(f"Error: {docs_src_dir} directory not found")
        return

    ensure_readme_files(docs_src_dir)

    # Generate and write SUMMARY.md
    summary_content = generate_summary(docs_src_dir)
    summary_path = docs_src_dir / "SUMMARY.md"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(summary_content)

    print(f"Generated SUMMARY.md at {summary_path}")


if __name__ == "__main__":
    main()
