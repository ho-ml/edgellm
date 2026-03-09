#!/bin/bash

# exit if error occurs
set -e

# move to project directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

# defaults
CONFIG="${CONFIG:-examples/configs/w4a16.yaml}"
MODEL="${MODEL:-meta-llama/Llama-3.2-1B}"

# run
python -m compressor.compress --config "$CONFIG" --model "$MODEL" "$@"
