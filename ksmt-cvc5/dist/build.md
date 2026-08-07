# cvc5 distribution for ksmt

`ksmt-cvc5-core` compiles against `cvc5-<version>.jar` and `ksmt-cvc5-native` ships one
`cvc5-native-<platform>-<version>.zip` per platform, consumed as a `flatDir` dependency.
The libraries in each zip are loaded, in order, by the matching
`KCvc5NativeLibraryLoader*` in `ksmt-cvc5-native`.

| Artifact | Contents | Origin |
| --- | --- | --- |
| `cvc5-<version>.jar` | `io.github.cvc5.*` classes, no natives | built from the fork (see below) |
| `cvc5-native-linux-x86-64-<version>.zip` | `libcvc5.so.1`, `libcvc5parser.so.1`, `libcvc5jni.so` | built from the fork |
| `cvc5-native-win-x86-64-<version>.zip` | `libcvc5jni.dll` | official cvc5 release |
| `cvc5-native-osx-arm64-<version>.zip` | `libcvc5jni.dylib` | official cvc5 release |

The jar must always come from the same cvc5 patch release as the native libraries:
`Kind` constants are generated from the C++ headers and new kinds are inserted mid-list,
so `Kind` ordinals shift between releases.

## The fork

Build is based on the cvc5 fork <https://github.com/Saloed/cvc5>, branch `ksmt-<version>`
(`ksmt` for the 1.3.0-era builds). It is a two-commit patch series on top of the upstream
release tag, and is refreshed by rebasing those two commits onto the new tag:

```shell
git rebase --onto cvc5-<new version> cvc5-<old version> <branch>
```

**`disable library load`** — patches the Java API:

* `AbstractPointer` and `IPointer` are made `public`. `KCvc5TermManager` uses
  `AbstractPointer` as a generic bound and calls `getPointer()`/`deletePointer()` on it.
* `Context`'s global pointer registry is disabled (`TRACK_POINTERS = false`). Upstream
  registers *every* `Term`/`Sort`/`Op` in a static map that is only drained by
  `Context.deletePointers()`. ksmt manages cvc5 objects per solver via `KCvc5TermManager`
  and never calls that JVM-global method, so the registry would retain every term ksmt
  ever creates.
* cvc5 does not load its native libraries from its own static initializers. ksmt ships
  the libraries in its own resource layout and loads them explicitly
  (`KCvc5NativeLibraryLoader`). This is belt-and-braces: `KCvc5Solver` also sets
  `-Dcvc5.skipLibraryLoad=true`, which upstream honours.

**`Fix build`** — adds `build_(linux|windows|mac).sh` and cmake fixes: libpoly is linked
statically even for a shared cvc5 build, `CMAKE_AR` is honoured when cross-compiling, and
the macOS deployment target is pinned.

## Building linux-x86-64

Built in an AlmaLinux 8 container, which provides a glibc (2.28) older than any supported
distribution together with a C++17 toolchain. Do **not** use the official Linux release
artifacts: they require glibc 2.33 and export ~1350 `std::` symbols, which interpose on
the C++ runtime of the other native solvers in the same JVM.

```shell
docker run --rm -v "$PWD:/work" -e "HOST_UID=$(id -u)" -e "HOST_GID=$(id -g)" \
    -w /work almalinux:8 bash /work/cvc5/build_linux.sh

cd linux-dist/dist
zip -X -9 "cvc5-native-linux-x86-64-<version>.zip" libcvc5jni.so libcvc5parser.so.1 libcvc5.so.1
```

with the fork checked out in `./cvc5`. See `build_linux.sh` in the fork for the full
recipe; the parts that matter are:

* `./configure.sh production --auto-download --ipo --no-static --no-cln --no-glpk
  --no-editline --java-bindings`
* `LDFLAGS="-static-libgcc -static-libstdc++ -Wl,--exclude-libs=ALL"`. The static linking
  keeps the libraries independent of the host C++ runtime; `--exclude-libs=ALL` is
  **required** to keep those statically linked symbols out of the dynamic symbol table,
  otherwise they interpose on the libstdc++ used by the Bitwuzla / Z3 / Yices libraries
  loaded into the same JVM and crash it with a SIGSEGV.
* GMP is built statically and separately, because for a shared cvc5 build cvc5 would
  otherwise produce a shared GMP that we would have to ship as a fourth library.
* The jar is built with JDK 8 — ksmt targets Java 8.
* Since 1.3.4 `libcvc5.so.1` and `libcvc5parser.so.1` are symlinks to the fully versioned
  files, so packaging must dereference them (`cp -L`).

## Windows x64 and macOS arm64

Taken from the official [cvc5 release](https://github.com/cvc5/cvc5/releases). Since
cvc5 1.3.2 the `cvc5-<platform>-java-api.jar` release assets contain a single,
self-contained JNI library under `cvc5-libs/<os>/<arch>/` with libcvc5, libcvc5parser,
GMP and the C++ runtime all linked in, so no post-processing is needed:

```shell
unzip -j "cvc5-Win64-x86_64-java-api.jar"  "cvc5-libs/windows/x86_64/cvc5jni.dll"
zip -X -9 "cvc5-native-win-x86-64-<version>.zip" libcvc5jni.dll   # note: renamed

unzip -j "cvc5-macOS-arm64-java-api.jar" "cvc5-libs/osx/aarch_64/libcvc5jni.dylib"
zip -X -9 "cvc5-native-osx-arm64-<version>.zip" libcvc5jni.dylib
```

Do **not** use the `-shared.zip` assets: their libraries reference each other through
`@rpath`/relative names, and on macOS they are ad-hoc code signed, so the install-name
rewriting the old fork-based build did would invalidate the signature.

Note that the Windows library links the UCRT, so it requires Windows 10 or newer.

## Verifying the result

Run these before committing new archives.

```shell
# The version is the one we intended to ship
strings -a libcvc5.so.1 | grep -m1 -E '^1\.[0-9]+\.[0-9]+$'

# No dependency on the system C++ runtime (expect 0)
objdump -T libcvc5.so.1 libcvc5parser.so.1 libcvc5jni.so | grep -c GLIBCXX

# Old glibc is enough
objdump -T libcvc5.so.1 libcvc5parser.so.1 libcvc5jni.so | grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1

# The static C++ runtime is not exported (expect a few hundred, not thousands)
nm -D --defined-only -C libcvc5.so.1 | grep -c 'std::'

# The jar is Java 8 and carries no natives
unzip -l cvc5-<version>.jar | grep -c cvc5-libs
```

For cvc5 1.3.4 this gives version `1.3.4`, `0` GLIBCXX references, max `GLIBC_2.25`,
`402`/`251`/`10` exported `std::` symbols, and `0` native entries in the jar.

Finally run the tests, including `:ksmt-test:test`, which loads several solvers into a
single JVM and is what catches the symbol interposition problem described above.

## Expected dynamic dependencies

To ensure the distribution is portable, verify the produced binaries have no dependencies
that might not be present on the user's machine.

### Linux x64

```shell
$ objdump -p libcvc5.so.1 libcvc5parser.so.1 libcvc5jni.so | grep -E 'SONAME|NEEDED'
libcvc5.so.1:        NEEDED libm.so.6, libc.so.6, ld-linux-x86-64.so.2
                     SONAME libcvc5.so.1
libcvc5parser.so.1:  NEEDED libcvc5.so.1, libm.so.6, libc.so.6, ld-linux-x86-64.so.2
                     SONAME libcvc5parser.so.1
libcvc5jni.so:       NEEDED libcvc5parser.so.1, libcvc5.so.1, libm.so.6, libc.so.6, ld-linux-x86-64.so.2
                     SONAME libcvc5jni.so
```

### Windows x64

```shell
$ objdump -p libcvc5jni.dll | grep 'DLL Name'
        DLL Name: api-ms-win-crt-{convert,environment,filesystem,heap,locale,math,
                                  multibyte,private,runtime,stdio,string,time,utility}-l1-1-0.dll
        DLL Name: KERNEL32.dll
```

No `libstdc++-6.dll`, `libgcc_s_seh-1.dll`, `libwinpthread-1.dll` and no sibling cvc5 DLL.

### MacOS aarch64

```shell
$ otool -L libcvc5jni.dylib
libcvc5jni.dylib:
        @rpath/libcvc5jni.dylib (compatibility version 0.0.0, current version 0.0.0)
        /usr/lib/libc++.1.dylib
        /usr/lib/libSystem.B.dylib
```

Both dependencies are absolute system paths and there is no `LC_RPATH`, so the `@rpath`
install name is never resolved — ksmt loads the library by absolute path.
