# Below specified build under linux (lib + jni bindings).
#
# To build under Windows, we have to add "--win64" option to ./configure.py (it picks up the cross-compile file).
# As host, the lib can be built under linux using cross-compilation tools with mingw-w64
# (Actually, the size of libbitwuzla.dll built under linux is terrible ~17 MB.
# Therefore, it's better to build bitwuzla in MSYS2 and Windows as host directly ~7 MB).
# As for jni bindings, we need MSYS2 with MinGW shell under Windows
#

JAVA_HOME="/usr/lib/jvm/default"  # or another path to Java 8

# https://github.com/bitwuzla/bitwuzla/commit/b655bc0cde570258367bf8f09a113bc7b95e46e9
git clone https://github.com/bitwuzla/bitwuzla
cd bitwuzla
git checkout b655bc0cde570258367bf8f09a113bc7b95e46e9

#
# Patch description
#
# 1. Added noexcept(false) in BitwuzlaAbortStream destructor (src/api/c/checks.h)
# because there is abortion callback is called and it can throw any (see abort_callback impl in bitwuzla_jni.cpp).
# 
# 2. Added extension functions (such as on sort/term dec external references, bitvector string/uint64 representation) 
# declarations (include/bitwuzla/c/bitwuzla.h) and definitions (src/api/c/bitwuzla.cpp)
#
# 3. Meson build files: gmp, bitblast_lib, bitvector_lib, local_search_lib - static linkage specified, 
# removed version of final .so lib (we will have libbitwuzla.so instead of libbitwuzla.so.0).
# Also added linker options for Windows build (static linkage with platform libs)
#

git apply ../bitwuzla_patch.patch

# build bitwuzla

# uncomment if you want to build with python as venv (meson and ninja are needed)
python -m venv ./venv
source venv/bin/activate
pip install meson
#

rm -r build
./configure.py release --shared
cd build && meson compile

# build JNI

cd ../..
rm -r build
mkdir build && cd build
cmake ..
make


# Done
# look for libbitwuzla.so in ./bitwuzla/build/src/libbitwuzla.so
# look for libbitwuzla_jni.so in ./build/libbitwuzla_jni.so
