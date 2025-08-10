#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import io
import os
import re
import sys
from collections import OrderedDict
from typing import Dict, List, Tuple


Section = str


def normalize_status(raw: str) -> str:
    raw = raw.strip().upper()
    mapping = {
        "FAILED": "fail",
        "ERROR": "error",
        "PASSED": "pass",
        "SKIPPED": "skip",
        "XFAIL": "xfail",
        "XPASS": "xpass",
        "XFAILED": "xfail",
        "XPASSED": "xpass",
        "CRASHED": "crashed",
    }
    return mapping.get(raw, raw.lower())


def display_status_upper(status: str) -> str:
    mapping = {
        "fail": "FAILED",
        "error": "ERROR",
        "pass": "PASSED",
        "skip": "SKIPPED",
        "xfail": "XFAIL",
        "xpass": "XPASS",
        "crashed": "ERROR",
    }
    return mapping.get(status, status.upper())


def derive_short_name(nodeid: str) -> str:
    # Convert e.g. tests/foo.py::test_all_models[param] -> test_all_models[param]
    if "::" in nodeid:
        return nodeid.split("::", 1)[1]
    return nodeid


def _shorten_message(message: str) -> str:
    if not message:
        return message
    # Drop verbose details that follow our join separator or markers
    # Keep only content before ' | ' or 'Detail:' if present
    for sep in [" | ", "Detail:"]:
        if sep in message:
            message = message.split(sep, 1)[0].strip()
    # Collapse whitespace
    message = re.sub(r"\s+", " ", message).strip()
    return message


def parse_failure_blocks(lines: List[str]) -> Dict[str, Tuple[str, str, str]]:
    # Returns mapping: short_name -> (status, message, full_nodeid_or_empty)
    results: Dict[str, Tuple[str, str, str]] = {}

    section_type: Section = ""  # "failures" or "errors" when applicable
    # Patterns
    section_failures_re = re.compile(r"^=+\s*FAILURES\s*=+\s*$")
    section_errors_re = re.compile(r"^=+\s*ERRORS\s*=+\s*$")
    header_re = re.compile(r"^_+\s+(?P<name>test[^\s]+)\s+_+\s*$")
    crashed_re = re.compile(r"CRASHED with signal\s+(?P<sig>\d+)")
    xpass_re = re.compile(r"\[XPASS(?:[^\]]*)\]\s*(?P<msg>.*)")
    xfail_re = re.compile(r"\[XFAIL(?:[^\]]*)\]\s*(?P<msg>.*)")
    running_re = re.compile(r"^Running\s+(?P<node>[^\s].+?::\S+)\s*$")

    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]

        # Reset section when a new top-level section header starts
        if line.startswith("==== "):
            section_type = ""

        if section_failures_re.match(line):
            section_type = "failures"
            i += 1
            continue
        if section_errors_re.match(line):
            section_type = "errors"
            i += 1
            continue

        m = header_re.match(line)
        if m:
            short_name = m.group("name").strip()
            # Collect block until next underscore header or next big section or EOF
            block_start = i + 1
            j = block_start
            while j < n:
                if (
                    header_re.match(lines[j])
                    or section_failures_re.match(lines[j])
                    or section_errors_re.match(lines[j])
                    or lines[j].startswith("==== ")
                ):
                    break
                j += 1
            block = lines[block_start:j]

            status: str | None = None
            message: str | None = None
            full_nodeid: str = ""

            # Try to find full nodeid from 'Running <node>' lines
            for bline in block:
                rm = running_re.match(bline.strip())
                if rm:
                    full_nodeid = rm.group("node").strip()
                    break

            # First, look for XPASS/XFAIL markers regardless of section
            for bline in block:
                xm = xpass_re.search(bline)
                if xm:
                    status = "xpass"
                    message = xm.group("msg").strip()
                    break
                xm = xfail_re.search(bline)
                if xm:
                    status = "xfail"
                    message = xm.group("msg").strip()
                    break

            # If no XPASS/XFAIL markers, determine status from section type
            if status is None:
                if section_type == "failures":
                    status = "fail"
                elif section_type == "errors":
                    status = "error"

            # Prefer explicit crash message if present
            if status in {"fail", "error"} or status is None:
                for bline in block:
                    cm = crashed_re.search(bline)
                    if cm:
                        message = bline.strip()
                        status = "crashed"
                        break

            if status in {"fail", "error", "crashed"}:
                if not message or status != "crashed":
                    # Collect lines that start with 'E' (pytest error summaries)
                    error_lines: List[str] = []
                    for bline in block:
                        stripped = bline.lstrip()
                        if (
                            stripped.startswith("E ")
                            or stripped == "E"
                            or stripped.startswith("E\t")
                        ):
                            error_lines.append(
                                stripped[2:].strip()
                                if stripped.startswith("E ")
                                else stripped.strip()
                            )
                    if error_lines and not message:
                        # Use only the first E-line as concise message
                        message = error_lines[0].strip()

                if not message:
                    # Fallback: look for lines mentioning common error types in block
                    for bline in reversed(block):
                        if any(
                            tok in bline
                            for tok in (
                                "AssertionError",
                                "RuntimeError",
                                "TypeError",
                                "ValueError",
                            )
                        ):
                            message = bline.strip()
                            break

                if not message:
                    message = "(no error summary found)"

            # If still no status determined (e.g., not in failures/errors and no xpass/xfail/crash), skip this header block
            if not status:
                i = j
                continue

            message = _shorten_message(message or "")

            # Only set if not already present; prefer earlier insertion order
            if short_name not in results:
                results[short_name] = (status, message, full_nodeid)

            i = j
            continue

        i += 1

    return results


def parse_short_summary(lines: List[str]) -> Dict[str, Tuple[str, str, str]]:
    # Returns mapping: short_name -> (status, message, full_nodeid)
    results: Dict[str, Tuple[str, str, str]] = {}
    start_re = re.compile(r"^=+\s*short test summary info\s*=+\s*$", re.IGNORECASE)
    # Require separator as ' - ' (spaces around hyphen) to avoid splitting nodeids containing '-' in params
    entry_re = re.compile(
        r"^(FAILED|ERROR|PASSED|SKIPPED|XFAIL|XPASS|XFAILED|XPASSED)\s+(.+?)(?:\s+-\s+(.*))?$"
    )

    i = 0
    n = len(lines)
    while i < n:
        if start_re.match(lines[i]):
            i += 1
            # Iterate until blank line or next section
            while i < n and not lines[i].startswith("==== ") and lines[i].strip() != "":
                m = entry_re.match(lines[i].rstrip())
                if m:
                    raw_status, node_or_path, maybe_msg = (
                        m.group(1),
                        m.group(2),
                        m.group(3),
                    )
                    status = normalize_status(raw_status)
                    text = node_or_path.strip()
                    # Only keep entries that reference a specific test (contain '::')
                    if "::" not in text:
                        i += 1
                        continue
                    short = derive_short_name(text)
                    msg = (maybe_msg or "").strip()
                    results[short] = (status, _shorten_message(msg), text)
                i += 1
        else:
            i += 1

    return results


def parse_progress_status(lines: List[str]) -> Dict[str, Tuple[str, str, str]]:
    # Capture verbose per-test status lines if present
    # Returns mapping: short_name -> (status, message, full_nodeid)
    results: Dict[str, Tuple[str, str, str]] = {}
    # Example: tests/test_all_models.py::test_all_models[param] PASSED
    prog_re = re.compile(r"^(.+::\S+)\s+(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)\b")
    for line in lines:
        m = prog_re.match(line.rstrip())
        if not m:
            continue
        nodeid = m.group(1).strip()
        status = normalize_status(m.group(2))
        short = derive_short_name(nodeid)
        results[short] = (status, "", nodeid)
    return results


def parse_log(lines: List[str]) -> OrderedDict:
    # Ordered by first appearance across parsing passes
    ordered: OrderedDict[str, Dict[str, str]] = OrderedDict()

    # First, collect progress and summary for authoritative status
    progress = parse_progress_status(lines)
    for name, (status, _msg, full) in progress.items():
        ordered.setdefault(name, {"status": status, "message": "", "full": full})

    summary = parse_short_summary(lines)
    for name, (status, message, full) in summary.items():
        if name in ordered:
            ordered[name]["status"] = status
            if message:
                ordered[name]["message"] = message
            if full:
                ordered[name]["full"] = full
        else:
            ordered[name] = {"status": status, "message": message, "full": full}

    # Then, fill in messages from failure/error blocks, but do not override PASSED/XPASS/SKIPPED
    blocks = parse_failure_blocks(lines)
    for name, (status, message, full) in blocks.items():
        if name in ordered:
            if ordered[name]["status"] in {"pass", "xpass", "skip", "xfail"}:
                if not ordered[name].get("message") and message:
                    ordered[name]["message"] = message
                if full and not ordered[name].get("full"):
                    ordered[name]["full"] = full
                continue
            ordered[name]["status"] = status
            if message:
                ordered[name]["message"] = message
            if full and not ordered[name].get("full"):
                ordered[name]["full"] = full
        else:
            ordered[name] = {"status": status, "message": message, "full": full}

    return ordered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Post-process pytest log to emit concise per-test summary with error messages."
    )
    parser.add_argument("logfile", help="Path to pytest log file")
    parser.add_argument(
        "--only",
        choices=["all", "failures"],
        default="all",
        help="Show all parsed tests or only failures/errors/crashes",
    )
    parser.add_argument(
        "--max-msg-len",
        type=int,
        default=300,
        help="Max characters of error message to show per test",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.logfile):
        print(f"Log file not found: {args.logfile}", file=sys.stderr)
        return 1

    with io.open(args.logfile, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    summary = parse_log(lines)

    for name, data in summary.items():
        status = data.get("status", "")
        message = data.get("message", "").strip()
        full = data.get("full") or name

        if args.only == "failures" and status not in {"fail", "error", "crashed"}:
            continue

        if not message and status in {"pass", "skip", "xfail", "xpass"}:
            message = ""

        if args.max_msg_len > 0 and len(message) > args.max_msg_len:
            message = message[: args.max_msg_len - 1] + "…"

        status_up = display_status_upper(status)
        if message:
            print(f"{status_up} {full} {message}")
        else:
            print(f"{status_up} {full}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
