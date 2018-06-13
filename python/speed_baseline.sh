###############################################################################
# This file is part of GALARIO:                                               #
# Gpu Accelerated Library for Analysing Radio Interferometer Observations     #
#                                                                             #
# Copyright (C) 2017-2018, Marco Tazzari, Frederik Beaujean, Leonardo Testi.  #
#                                                                             #
# This program is free software: you can redistribute it and/or modify        #
# it under the terms of the Lesser GNU General Public License as published by #
# the Free Software Foundation, either version 3 of the License, or           #
# (at your option) any later version.                                         #
#                                                                             #
# This program is distributed in the hope that it will be useful,             #
# but WITHOUT ANY WARRANTY; without even the implied warranty of              #
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.                        #
#                                                                             #
# For more details see the LICENSE file.                                      #
# For documentation see https://mtazzari.github.io/galario/                   #
###############################################################################

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
