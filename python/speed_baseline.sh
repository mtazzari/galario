#! /bin/bash

# output file name is first arg
if [ -z "$1" ]; then
    output="baseline_$(git rev-parse --short HEAD)_$HOSTNAME.log"
else
    output=$1
fi

openmp_threads="1 2 4 8 16 32"
threads_per_block="8 16 32"
cycles=5
sizes="512 1024 2048 4096 8192 16384"

python speed_benchmark.py --output_header --output=$output;

cmd="python speed_benchmark.py --timing --cycles=$cycles --output=$output"

## GPU
for s in $sizes; do
    for tpb in ${threads_per_block}; do
	$cmd --size=$s --gpu --tpb=$tpb;
    done
done

## CPU
for s in $sizes; do
    for nthreads in ${openmp_threads}; do
	export OMP_NUM_THREADS=$nthreads;
	$cmd --size=$s;
    done
done


