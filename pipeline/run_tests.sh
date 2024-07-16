#!/bin/bash

# Directory containing the configuration files
CONFIG_DIR="test_configs/$1"

# Loop over each configuration file in the directory
for CONFIG_FILE in "$CONFIG_DIR"/*
do
    # Execute the Python script with the current configuration file
    python3 perform_experiment.py --config "$CONFIG_FILE"
done