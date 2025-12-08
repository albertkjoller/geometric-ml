#!/bin/sh
### DEFINE JOB PARAMETERS HERE...

### Define parameters to sweep
noise_types=("ambient" "brownian")
noise_intensities=(0.00001 0.0001 0.001 0.01 0.1 1.0)

### SET SAVE PATH
SAVEPATH=/work3/s194253/projects/geometric_noise/results

### Load modules
module load cuda/12.4
module load python3/3.11.9

### Activate environment and change path
source /work3/s194253/envs/geo/bin/activate
cd /work3/s194253/projects/geometric_noise

### Train autoencoder with given seed
echo "Training autoencoder with seed $SEED"
python train_mnist_autoencoder.py --seed $SEED --save-folder $SAVEPATH --device cuda --subsample-fraction 1.0

### Train classifier without noise
echo "Training noise-free classifier with seed $SEED"
python train_mnist_classifier.py --autoencoder-checkpoint "mnist_autoencoder/fraction=1.0_sigmoid=False" --method standard --noise-intensity 0 --seed $SEED --save-folder $SAVEPATH --device cuda --subsample-fraction $FRACTION
### Train classifier on reconstructions
echo "Training classifier on reconstructions with seed $SEED"
python train_mnist_classifier.py --autoencoder-checkpoint "mnist_autoencoder/fraction=1.0_sigmoid=False" --method reconstructed --noise-intensity 0 --seed $SEED --save-folder $SAVEPATH --device cuda --subsample-fraction $FRACTION

# Run experiments
for noise_intensity in "${noise_intensities[@]}"; do
  for noise_type in "${noise_types[@]}"; do
    echo "Training classifier with $noise_type noise of intensity $noise_intensity and seed $SEED"
    python train_mnist_classifier.py --autoencoder-checkpoint "mnist_autoencoder/fraction=1.0_sigmoid=False" --method $noise_type --noise-intensity $noise_intensity --seed $SEED --save-folder $SAVEPATH --device cuda --subsample-fraction $FRACTION
  done
done