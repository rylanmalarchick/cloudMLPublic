#!/usr/bin/env bash
#
# Run vision baselines training in background with logging
# Estimated time: 20-30 hours
#

set -e

# Navigate to project root
cd "$(dirname "$0")/../.."

# Create output directory
mkdir -p outputs/vision_baselines/logs

# Log file with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="outputs/vision_baselines/logs/training_${TIMESTAMP}.log"
PIDFILE="outputs/vision_baselines/logs/training.pid"

echo "Starting vision baselines training..."
echo "Logfile: $LOGFILE"
echo "PID file: $PIDFILE"

# Run in background with nohup
nohup ./venv/bin/python3 src/cbh_retrieval/vision_baselines.py > "$LOGFILE" 2>&1 &

# Save PID
echo $! > "$PIDFILE"

echo "Training started with PID $(cat $PIDFILE)"
echo "Monitor progress with: tail -f $LOGFILE"
echo "Stop training with: kill \$(cat $PIDFILE)"
