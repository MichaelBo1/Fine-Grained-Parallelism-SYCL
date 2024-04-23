#!/bin/bash

#SBATCH --partition=ug-gpu-small
#SBATCH --ntasks=1
#SBATCH --time=00:40:00
#SBATCH --output=./slurm-reports/%j.out

#SBATCH --gres=gpu:ampere:1
#SBATCH --nodelist=gpu12

output_file="timings_a100.csv"

cd $DIR
module load cuda/11.5 llvm-clang
# clang++ -Wall -fsycl -fsycl-targets=nvptx64-nvidia-cuda $FILE
pwd

if [ ! -s "${output_file}" ]; then
    echo "Event,ExecTime(ms),VectorSize,WorkGroupSize" >> $output_file
fi

for ((i = 10; i < 30; i++))
do
    vector_size=$((2**$i))
    ./${FILE} $vector_size
    for ((j=0; j < ${ITERS}; j++))
    do
        ./${FILE} $vector_size >> $output_file
    done
done
echo "Completed run on A100 for ${ITERS} iterations for ${FILE}"
current_date_time="`date "+%Y-%m-%d %H:%M:%S"`";
echo $current_date_time;