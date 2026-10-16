#!/usr/bin/env bash
set -euo pipefail

here=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
out=$(mktemp -d)

docker build \
  --build-arg BASE_REGISTRY="${BASE_REGISTRY:-docker.io}" \
  --target out \
  --output "type=local,dest=$out" \
  -f "$here/Dockerfile.bitwuzla-linux64" \
  "$here"

cp "$out/bitwuzla-linux64-1.0.zip" "$here/bitwuzla-linux64-1.0.zip"
echo "libbitwuzla.so sha256: $(cat "$out/libbitwuzla.so.sha256")"
rm -rf "$out"
