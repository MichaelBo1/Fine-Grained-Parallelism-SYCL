#!/bin/bash

# --------------
# Change nodelist and gres for different gpus
# --------------

#SBATCH --job-name=basic_usm_add
#SBATCH --output=basic_usm_add.out
#SBATCH --gres=gpu
#SBATCH --partition=ug-gpu-small
#SBATCH --ntasks=1
#SBATCH --nodelist=gpu4
#SBATCH --time=00:20:00

iterations=30
output_file="timings.txt"

# Execution
module load cuda/11.5 llvm-clang
clang++ -Wall -fsycl -fsycl-targets=nvptx64-nvidia-cuda vector_add_usm.cpp -o va_usm.exe

echo "Event,ExecTime(ms),VectorSize,Device" >> $output_file

for ((i=0; i<$iterations; i++))
do
    echo "$i"
    srun ./va_usm.exe >> $output_file
done
