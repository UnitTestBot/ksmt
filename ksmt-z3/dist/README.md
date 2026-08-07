# Z3 native libraries for linux-x86-64

`z3-native-linux-x86-64-<version>.zip` contains `libz3.so` and `libz3java.so` (at the archive root)
and is consumed by `ksmt-z3-native` as a `flatDir` dependency.

Unlike every other platform, these libraries are **not** taken from the
[Z3 GitHub release](https://github.com/Z3Prover/z3/releases). The released `x64-glibc-*` package is
built against a recent glibc (2.39 for Z3 5.0.0) and does not run on older distributions, including
the `ubuntu-22.04` CI runner. Instead, we build Z3 ourselves against an old glibc and link the C++
runtime statically.

Every other platform (`x64-win`, `x64-osx-*`, `arm64-osx-*`, `arm64-glibc-*`) is downloaded from the
Z3 release by `mkZ3ReleaseDownloadTask`, so only linux-x86-64 needs the procedure below.

## Building

The build runs in an AlmaLinux 8 container: it provides glibc 2.28 (older than any supported
distribution) together with a C++20 capable toolchain, which Z3 requires since 5.0.0
(it uses `<format>` and `<span>`, so GCC 13+ is a hard requirement).

```bash
Z3_VERSION=5.0.0

# Z3 sources
curl -sL -o z3.tar.gz "https://github.com/Z3Prover/z3/archive/refs/tags/z3-$Z3_VERSION.tar.gz"
tar xzf z3.tar.gz    # -> z3-z3-$Z3_VERSION

docker run --rm -v "$PWD:/work" -e "Z3_VERSION=$Z3_VERSION" \
    -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" almalinux:8 bash /work/build-z3-linux-x86-64.sh

cd z3-linux-dist && zip -X -9 "../z3-native-linux-x86-64-$Z3_VERSION.zip" libz3.so libz3java.so
```

with `build-z3-linux-x86-64.sh`:

```bash
#!/bin/bash
set -euo pipefail

dnf -y install --setopt=install_weak_deps=False \
    gcc-toolset-14 cmake make java-17-openjdk-devel python3

source /opt/rh/gcc-toolset-14/enable
export JAVA_HOME=$(dirname $(dirname $(readlink -f $(which javac))))

cd "/work/z3-z3-$Z3_VERSION"

# The CMake build gives libz3 a versioned SONAME (e.g. libz3.so.5.0) and makes libz3java
# depend on that name, but the distribution ships a flat libz3.so. Drop VERSION/SOVERSION
# to keep the SONAME equal to the file name.
python3 - <<'EOF'
p = 'src/CMakeLists.txt'
s = open(p).read()
patched = s.replace("""  VERSION ${Z3_VERSION}
  SOVERSION ${Z3_VERSION_MAJOR}.${Z3_VERSION_MINOR})""", "  POSITION_INDEPENDENT_CODE ON)")
assert patched != s, "SOVERSION patch did not apply"
open(p, 'w').write(patched)
EOF

cmake -S . -B build-linux \
    -DCMAKE_BUILD_TYPE=Release \
    -DZ3_BUILD_JAVA_BINDINGS=TRUE \
    -DZ3_BUILD_LIBZ3_SHARED=TRUE \
    -DZ3_BUILD_EXECUTABLE=FALSE \
    -DZ3_BUILD_TEST_EXECUTABLES=FALSE \
    -DZ3_INCLUDE_GIT_HASH=FALSE \
    -DZ3_INCLUDE_GIT_DESCRIBE=FALSE \
    -DCMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL" \
    -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -Wl,--exclude-libs,ALL"

cmake --build build-linux --target libz3 z3java -j "$(nproc)"

mkdir -p /work/z3-linux-dist
cp build-linux/libz3.so build-linux/libz3java.so /work/z3-linux-dist/
chown -R "$HOST_UID:$HOST_GID" /work/z3-linux-dist "/work/z3-z3-$Z3_VERSION"
```

### Why these linker flags

* `-static-libstdc++ -static-libgcc` — the resulting libraries must not depend on the libstdc++
  of the machine that runs ksmt.
* `-Wl,--exclude-libs,ALL` — **required**. Without it the statically linked libstdc++ symbols stay
  in the dynamic symbol table (~4500 exported `std::` symbols). They then interpose on the
  libstdc++ used by the Bitwuzla / cvc5 / Yices native libraries loaded into the same JVM,
  which crashes the JVM with a SIGSEGV inside `libz3.so`.

## Verifying the result

Run these checks before committing a new archive — a mistake here produces libraries that either
crash at runtime or silently keep the previous Z3 version.

```bash
cd z3-linux-dist

# 1. The version is the one we intended to build
strings -a libz3.so | grep -m1 -E '^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$'

# 2. No dependency on the system C++ runtime (expect 0)
objdump -T libz3.so libz3java.so | grep -c GLIBCXX

# 3. Old glibc is enough (expect something well below the oldest supported distribution)
objdump -T libz3.so libz3java.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1

# 4. The static C++ runtime is not exported (expect ~90, not ~4500)
nm -D --defined-only -C libz3.so | grep -c 'std::'

# 5. SONAMEs match the file names, and libz3java depends on plain libz3.so
objdump -p libz3.so     | grep -E 'SONAME|NEEDED'
objdump -p libz3java.so | grep -E 'SONAME|NEEDED'
```

For Z3 5.0.0 this gives: version `5.0.0.0`, `0` GLIBCXX references, max `GLIBC_2.26`,
`91` exported `std::` symbols, `SONAME libz3.so` / `SONAME libz3java.so`.

Finally run the tests, including `:ksmt-test:test`, which loads several solvers into a single JVM
and is what catches the symbol interposition problem described above.
