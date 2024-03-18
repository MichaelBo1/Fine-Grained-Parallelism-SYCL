#!/bin/bash

#SBATCH --partition=ug-gpu-small
#SBATCH --ntasks=1
#SBATCH --time=00:40:00

#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu12

output_file="timings.csv"

cd $DIR
module load cuda/11.5 llvm-clang
clang++ -Wall -fsycl -fsycl-targets=nvptx64-nvidia-cuda $FILE
pwd

if [ ! -s "${output_file}" ]; then
    echo "Event,ExecTime(ms),VectorSize,Device" >> $output_file
fi

for ((i=0; i<${ITERS}; i++))
do
    ./a.out >> $output_file
done
echo "Completed run on A100 for ${ITERS} iterations for ${FILE}"