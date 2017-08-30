#! /bin/bash

# stop on first error
set -e

# output file name is first arg
if [ -z "$1" ]; then
    output="baseline_$(git rev-parse --short HEAD)_$HOSTNAME.log"
else
    output=$1
fi

# recreate empty file
:> $output
cycles=10

openmp_threads="1 2 4 8 16 32 48 64"
threads_per_block="32 64 128 256 512 1024"
sizes="512 1024 2048 4096 8192 16384"

# quick testing
# openmp_threads="1 16"
# threads_per_block="128 256"
# sizes="512 1024"

python speed_benchmark.py --output_header --output=$output;

cmd="python speed_benchmark.py --timing --cycles=$cycles --output=${output} --no-verbose"

for s in $sizes; do
    $cmd --size=$s --gpu --tpb ${threads_per_block} --ompnthreads ${openmp_threads} >> $output 2>&1;
done
