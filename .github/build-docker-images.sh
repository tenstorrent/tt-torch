#!/bin/bash
# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -e

REPO=tenstorrent/tt-torch
BASE_IMAGE_NAME=ghcr.io/$REPO/tt-torch-base-ubuntu-22-04
CI_IMAGE_NAME=ghcr.io/$REPO/tt-torch-ci-ubuntu-22-04
CI_BUILDWHEEL_IMAGE_NAME=ghcr.io/$REPO/tt-torch-manylinux-amd64

# Compute the hash of the Dockerfile
DOCKER_TAG=$(./.github/get-docker-tag.sh)
echo "Docker tag: $DOCKER_TAG"

build_and_push() {
    local image_name=$1
    local dockerfile=$2

    if docker manifest inspect $image_name:$DOCKER_TAG > /dev/null; then
        echo "Image $image_name:$DOCKER_TAG already exists"
    else
        echo "Building image $image_name:$DOCKER_TAG"
        docker build \
            --progress=plain \
            --build-arg FROM_TAG=$DOCKER_TAG \
            -t $image_name:$DOCKER_TAG \
            -f $dockerfile .

        echo "Pushing image $image_name:$DOCKER_TAG"
        docker push $image_name:$DOCKER_TAG
    fi
}

build_and_push $BASE_IMAGE_NAME .github/Dockerfile.base
build_and_push $CI_IMAGE_NAME .github/Dockerfile.ci
build_and_push $CI_BUILDWHEEL_IMAGE_NAME .github/Dockerfile.cibuildwheel

echo "All images built and pushed successfully"
echo "CI_IMAGE_NAME:"
echo $CI_IMAGE_NAME:$DOCKER_TAG
