#!/bin/bash

# Define the base command
BASE_CMD="python training/main.py -d training/train/ --batch-size 24 --test-batch-size 16 -lr 1e-4 --cuda -e 100 --ae"

# Define the lambda values to sweep
LAMBDAS=(1e-3 7e-4 2e-3 3e-3 5e-4)

# Loop over each lambda value
for LAMBDA in "${LAMBDAS[@]}"; do
    # Define a unique checkpoint filename for each lambda value
    CHECKPOINT="checkpoint_lambda_${LAMBDA//./}.pth.tar"
    
    # Construct the full command
    if [ -f "$CHECKPOINT" ]; then
        CMD="$BASE_CMD --lambda $LAMBDA --checkpoint $CHECKPOINT --filename ${CHECKPOINT%.pth.tar}"
    else
        CMD="$BASE_CMD --lambda $LAMBDA --filename ${CHECKPOINT%.pth.tar}"
    fi
    
    # Run the command
    echo "Running: $CMD"
    $CMD
done
