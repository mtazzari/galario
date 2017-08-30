#! /bin/bash

# stop on first error
set -e

# output file name is first arg
if [ -z "$1" ]; then
    output="baseline_$(git rev-parse --short HEAD)_$HOSTNAME.txt"
else
    output=$1
fi

openmp_threads="1 2 4 6 8 10 12"
threads_per_block="8 16 32"
sizes="512 1024 2048 4096 8192 16384"
cycles=20

# quick testing
#openmp_threads="6"
#threads_per_block="32"
#sizes="4096"
#cycles=10

# profile
output1=profile_${output}
:> $output1 # recreate empty file

python speed_benchmark.py --output_header --output=$output1;

cmd="python speed_benchmark.py --timing --cycles=$cycles --output=${output1} --no-verbose"

for s in $sizes; do
    $cmd --size=$s --gpu --tpb ${threads_per_block} --ompnthreads ${openmp_threads} >> $output1 2>&1;
done

# image
output2="image_${output}"
:> $output2 # recreate empty file

python speed_benchmark.py --output_header --output=${output2} --image;

cmd="python speed_benchmark.py --timing --cycles=$cycles --output=${output2} --no-verbose --image"

for s in $sizes; do
    $cmd --size=$s --gpu --tpb ${threads_per_block} --ompnthreads ${openmp_threads} >> $output2 2>&1;
done
