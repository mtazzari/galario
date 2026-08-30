# Reproduce the Linux CPU installation with pip-managed Python dependencies.
# CUDA is intentionally disabled here: a CUDA build requires an NVIDIA toolkit
# and is covered by the same CMake path when GALARIO_CHECK_CUDA is enabled.
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    OMP_NUM_THREADS=2 \
    PYTHONPATH=/opt/galario/lib/python3.10/site-packages \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        libfftw3-dev \
        python3 \
        python3-dev \
        python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir \
        'Cython>=3' \
        numpy \
        pytest \
        pytest-cov \
        scipy

WORKDIR /src
COPY . .

RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/galario \
        -DGALARIO_CHECK_CUDA=OFF \
    && cmake --build build --parallel \
    && cmake --install build \
    && python3 -c 'import galario; print(galario.__file__)' \
    && ctest --test-dir build --output-on-failure
