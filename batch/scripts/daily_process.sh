#!/bin/bash

# Directory where file will be created
DIR="/workspaces/codespaces-demo/batch/output"

# Timestamp format: YYYYMMDD_HHMMSS
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# File name
FILENAME="DLY_${TIMESTAMP}.txt"

# Create file
touch "$DIR/$FILENAME"

echo "File created: $DIR/$FILENAME"