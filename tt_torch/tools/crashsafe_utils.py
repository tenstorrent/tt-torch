# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
crashsafe_suffix = "_crashsafe.json"

from json2xml import json2xml
from json2xml.utils import readfromurl, readfromstring, readfromjson
import sys

# Convert a JSON file to XML


def reconstruct_junitxml_from_crashsafe(junitxml_path: str):
    # from here, how can the name of the file be determined?
    # assume it's passed in from the runner / eg. using the shared strings

    crashsafe_path = f"{junitxml_path}{crashsafe_suffix}"
    junitxml_path = f"{junitxml_path}_regenerated.xml"
    data = readfromjson(crashsafe_path)

    with open(junitxml_path, "w+") as f:
        f.write(json2xml.Json2xml(data).to_xml())


reconstruct_junitxml_from_crashsafe(sys.argv[1])
