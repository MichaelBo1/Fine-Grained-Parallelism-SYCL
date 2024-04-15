#!/bin/bash

#SBATCH --partition=ug-gpu-small
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --output=./slurm-reports/%j.out

#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu12

output_file="timings_a100.csv"

module load cuda/11.5 llvm-clang
# clang++ -Wall -fsycl -fsycl-targets=nvptx64-nvidia-cuda vector_add_gsq.cpp

if [ ! -s "${output_file}" ]; then
    echo "Event,ExecTime(ms),VectorSize" >> $output_file
fi

for ((j=10; j < 30; j++))
do
    ./${j}
    for ((i=0; i < 30; i++))
        do
            ./${j} >> $output_file
        done
done

echo "Completed run on A100 for 30 iterations"
current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;