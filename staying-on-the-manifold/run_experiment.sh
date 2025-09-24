#!/bin/bash

# Define parameters to sweep
noise_types=("ambient" "tangent" "geodesic" "brownian")
noise_intensities=(0.00001 0.0001 0.001 0.01 0.1)
seeds=(0 1 2 3 4)

MANIFOLD_NAME="$1"  # Pass the manifold name as the first argument, e.g., "swiss-roll" or "sphere"
echo "Running experiments on manifold: $MANIFOLD_NAME"

for seed in "${seeds[@]}"; do
  # Run the standard network
  python regression.py --noise-intensity 0 --noise-type none --seed "$seed" --manifold-name "$MANIFOLD_NAME"

    # Run experiments
    for noise_intensity in "${noise_intensities[@]}"; do
    for noise_type in "${noise_types[@]}"; do
        python regression.py --noise-intensity "$noise_intensity" --noise-type "$noise_type" --seed "$seed" --manifold-name "$MANIFOLD_NAME"
    done
    done
done

## To make it runnable:
## chmod +x run_experiment.sh
## ./run_experiment.sh