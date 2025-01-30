#!/bin/bash

# Set the parent directory (change this to your actual directory)
PARENT_DIR="tests/models"

# Loop through all subdirectories within the parent directory
for subdir in "$PARENT_DIR"/*/; do
    if [ -d "$subdir" ]; then
        echo "Running pytest in $subdir"
        pytest -v "$subdir"
    fi
done