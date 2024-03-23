#apt-get install -y autoconf gperf

export CC=oa64-clang
export CXX=oa64-clang++

autoconf

export CFLAGS="-O3"
export CXXFLAGS="-O3"
export CPPFLAGS="-I$(realpath libgmp/dist/include)"
export LIBS=$(realpath libgmp/dist/lib/libgmp.a)
export LDFLAGS="-L$(realpath libgmp/dist/lib)"

./configure --enable-thread-safety --disable-mcsat --host=arm64-apple-darwin20.2 \
    --prefix=$(realpath dist-mac) \
    --with-pic-gmp=$(realpath libgmp/dist/lib/libgmp.a) \
    --with-pic-gmp-include-dir=$(realpath libgmp/dist/include)

make MODE=release ARCH=arm64-apple-darwin20.2 POSIXOS=darwin show-details dist install

llvm-install-name-tool-14 -id libyices.2.dylib dist-mac/lib/libyices.2.dylib
