#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Calculate hash for docker image tag.
# The hash is based on the MLIR docker tag  and the hash of the Dockerfile(s).

# Exit immediately if a command exits with a non-zero status
set -e

# Function to checkout tt-xla at the known TT_XLA_VERSION commit
checkout_tt_xla() {
    local tt_xla_version="$1"
    local tt_xla_path="$2"

    if [ ! -d "$tt_xla_path" ]; then
        git clone https://github.com/tenstorrent/tt-xla.git "$tt_xla_path" --quiet
    fi

    cd "$tt_xla_path"
    git fetch --quiet
    git checkout "$tt_xla_version" --quiet
    cd - > /dev/null
}

MLIR_DOCKER_TAG=$(
    # Read tt-mlir version from tt-xla's third_party/CMakeLists.txt
    # Clone tt-mlir version to tmp/third_party/tt-mlir
    # Get the MLIR docker tag

    # Note - third_party/tt-xla is cloned at the commit specified by the TT_XLA_VERSION in third_party/CMakeLists.txt
    #   when **running the cmake build**. In order to get the tt-mlir version pre-build, we can manually clone tt-xla
    #   at the specified version and read its contents

    TT_MLIR_PATH=tmp/third_party/tt-mlir
    TT_XLA_PATH=tmp/third_party/tt-xla

    # Extract TT_MLIR_VERSION from third_party/CMakeLists.txt if set, either by source modification
    # or application of TT_MLIR_OVERRIDE in build-image.yml
    TT_MLIR_VERSION=$(grep -oP 'set\(TT_MLIR_VERSION "\K[^"]+' third_party/CMakeLists.txt || echo "")

    if [ -n "$TT_MLIR_VERSION" ]; then
        : # TT_MLIR_VERSION is directly set in third_party/CMakeLists.txt due to override or source modification
    else
        # Extract TT_MLIR_VERSION from tt-xla's CMakeLists.txt
        TT_XLA_VERSION=$(grep -oP 'set\(TT_XLA_VERSION "\K[^"]+' third_party/CMakeLists.txt || echo "")

        if [ -z "$TT_XLA_VERSION" ]; then
            exit 1
        fi

        checkout_tt_xla "$TT_XLA_VERSION" "$TT_XLA_PATH"

        TT_MLIR_CMAKE_FILE="$TT_XLA_PATH/third_party/CMakeLists.txt"

        if [ ! -f "$TT_MLIR_CMAKE_FILE" ]; then
            exit 1
        fi

        TT_MLIR_VERSION=$(grep -oP 'set\(TT_MLIR_VERSION "\K[^"]+' "$TT_MLIR_CMAKE_FILE" || echo "")

        if [ -z "$TT_MLIR_VERSION" ]; then
            exit 1
        fi
    fi

    # Clone tt-mlir repository if needed
    if [ ! -d $TT_MLIR_PATH ]; then
        git clone https://github.com/tenstorrent/tt-mlir.git $TT_MLIR_PATH --quiet
    fi

    cd $TT_MLIR_PATH
    git fetch --quiet
    git checkout $TT_MLIR_VERSION --quiet

    if [ -f ".github/get-docker-tag.sh" ]; then
        .github/get-docker-tag.sh
    else
        echo "default-tag"
    fi
)
DOCKERFILE_HASH=$( (cat .github/Dockerfile.base .github/Dockerfile.ci | sha256sum) | cut -d ' ' -f 1)
COMBINED_HASH=$( (echo $DOCKERFILE_HASH $MLIR_DOCKER_TAG | sha256sum) | cut -d ' ' -f 1)
echo deezt-$COMBINED_HASH
