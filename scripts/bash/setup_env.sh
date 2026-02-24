#!/bin/bash

LOG_DIR="outputs/logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/setup_env_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting environment setup..."
echo "Log file: $LOG_FILE"

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete."