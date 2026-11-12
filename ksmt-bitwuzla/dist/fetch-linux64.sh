#!/usr/bin/env bash
set -euo pipefail

tag=${1:?usage: fetch-linux64.sh <release tag of metametamoon/bitwuzla>}
here=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
url="https://github.com/metametamoon/bitwuzla/releases/download/$tag"
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

curl -fsSL -o "$work/libbitwuzla.so" "$url/libbitwuzla.so"
curl -fsSL -o "$work/libbitwuzla.so.sha256" "$url/libbitwuzla.so.sha256"
(cd "$work" && sha256sum -c --quiet libbitwuzla.so.sha256)

python3 - "$here/bitwuzla-linux64-1.0.zip" "$work" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extract("libbitwuzla_jni.so", sys.argv[2])
PY

nm -D --undefined-only "$work/libbitwuzla_jni.so" | awk '{ print $NF }' | grep -E '^(bzla|bitwuzla)_' | sort -u > "$work/needed"
nm -D --defined-only "$work/libbitwuzla.so" | awk '{ print $NF }' | sort -u > "$work/defined"
missing=$(comm -23 "$work/needed" "$work/defined")
if [ -n "$missing" ]; then
  echo "libbitwuzla.so lacks symbols that libbitwuzla_jni.so imports:" >&2
  echo "$missing" >&2
  exit 1
fi

python3 - "$work" "$here/bitwuzla-linux64-1.0.zip" <<'PY'
import os, sys, zipfile
work, target = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(target + ".new", "w", zipfile.ZIP_DEFLATED) as z:
    for name in ("libbitwuzla.so", "libbitwuzla_jni.so"):
        info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
        info.external_attr = 0o755 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(os.path.join(work, name), "rb") as f:
            z.writestr(info, f.read())
os.replace(target + ".new", target)
PY

cut -d' ' -f1 "$work/libbitwuzla.so.sha256"
