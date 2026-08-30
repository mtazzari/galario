#!/usr/bin/env sh
# Build the Linux/amd64 Conda installation test image from the repository root.
set -eu

image_tag=${1:-galario-linux-conda-test:local}
platform=${DOCKER_PLATFORM:-linux/amd64}
exec docker build --quiet --platform "$platform" --file Dockerfile.conda --tag "$image_tag" .
