#!/bin/bash

LOG_DIR="outputs/logs"
mkdir -p $LOG_DIR

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/setup_env_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "Starting environment setup..."
echo "Log file: $LOG_FILE"

if [ -d "/scratch/$USER" ]; then
    VENV_DIR="/scratch/$USER/.venv"
else
    VENV_DIR=".venv"
fi

if [ -L ".venv" ] || [ -d ".venv" ]; then
    echo "Removing existing virtual environment..."
    rm -rf .venv "$VENV_DIR"
fi

echo "Creating virtual environment in $VENV_DIR..."
python3 -m venv "$VENV_DIR"
if [ "$VENV_DIR" != ".venv" ]; then
    ln -s "$VENV_DIR" .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete."