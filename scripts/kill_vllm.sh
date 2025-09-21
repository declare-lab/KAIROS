#!/bin/bash

# This script finds and kills the vLLM server process and all its children.

# The patterns to identify vLLM server processes
PROCESS_PATTERNS=(
    "from multiprocessing.spawn import spawn_main"
    "vllm.entrypoints.openai.api_server"
    # "eval_mas"
    "trl vllm-serve"
)

echo "Searching for vLLM server processes..."

# Use pgrep to find the PIDs of all matching processes
MAIN_PIDS=""
for pattern in "${PROCESS_PATTERNS[@]}"; do
    echo "Searching for pattern: $pattern"
    PIDS=$(pgrep -f "$pattern")
    if [ -n "$PIDS" ]; then
        echo "Found processes with pattern '$pattern': $PIDS"
        MAIN_PIDS="$MAIN_PIDS $PIDS"
    fi
done

# Kill the main processes and all their children
if [ -n "$MAIN_PIDS" ]; then
    echo "Killing vLLM server processes and all their children..."
    for pid in $MAIN_PIDS; do
        # Get all child PIDs
        CHILD_PIDS=$(pstree -p $pid | grep -o '([0-9]\+)' | grep -o '[0-9]\+')
        
        # Kill all child processes
        for child_pid in $CHILD_PIDS; do
            kill -9 $child_pid 2>/dev/null
        done
        
        # Kill the main process
        kill -9 $pid 2>/dev/null
    done
    echo "vLLM server processes terminated."
else
    echo "No vLLM server processes found."
fi



