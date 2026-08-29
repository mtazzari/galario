#!/usr/bin/env sh
# Build the Linux CPU installation test image from the repository root.
set -eu

image_tag=${1:-galario-linux-test:local}
platform=${DOCKER_PLATFORM:-linux/amd64}
exec docker build --quiet --platform "$platform" --tag "$image_tag" .
