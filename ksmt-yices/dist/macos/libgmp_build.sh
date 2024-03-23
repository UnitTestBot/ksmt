#wget https://ftp.gnu.org/gnu/gmp/gmp-6.3.0.tar.xz && tar Jxf gmp-6.3.0.tar.xz
cd gmp-6.3.0

export CC=oa64-clang
export CXX=oa64-clang++
export CFLAGS="-O3"
export CXXFLAGS="-O3"

./configure --host=arm64-apple-darwin20.2 --prefix=$(realpath ../dist) \
    --enable-static=yes --enable-shared=no --enable-cxx --with-pic

make -j$(nproc)
make install
