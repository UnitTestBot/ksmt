YICES_VERSION="2.6.4"

export MACOSX_DEPLOYMENT_TARGET=11.1
#apt-get install -y openjdk-8-jdk
#omp install openjdk11

export CC=oa64-clang
export CXX=oa64-clang++

JAVAC="/usr/bin/javac"
JAVA_HOME="/usr/local/osxcross/macports/pkgs/opt/local/Library/Java/JavaVirtualMachines/openjdk11/Contents/Home"
CPPFLAGS="-I $JAVA_HOME/include -I $JAVA_HOME/include/darwin -I $(realpath ../dist-mac/include) -I $(realpath ../libgmp/dist/include)"
CXXFLAGS="-fpermissive -g -fPIC -O3 -stdlib=libc++"
export LIBS="$(realpath ../libgmp/dist/lib/libgmp.a) $(realpath ../dist-mac/lib/libyices.2.dylib)"
export LDFLAGS="-L$(realpath ../libgmp/dist/lib) -L$(realpath ../dist-mac/lib/)"

YICES_2_JAVA_LIB_NAME="libyices2java.dylib"

rm -rf yices2_java_bindings

# https://github.com/SRI-CSL/yices2_java_bindings/commit/d9858e540425072443830d2638db5ffdc8c92cd1
git clone https://github.com/SRI-CSL/yices2_java_bindings
cd yices2_java_bindings
git checkout d9858e5

#
# Patch description
#
# Bindings native part:
# 1. Fix cardinality check in `expand function`.
# 2. Provide methods to extract values of product/sum components
#
# Bindings java part:
# 1. Add the corresponding methods and classes to interact
# with native product/sum value extraction methods
# 2. Disable default native library load
#
git apply ../yices-bindings-patch.patch


rm -rf build
mkdir build
cd build

cp ../src/main/java/com/sri/yices/yicesJNI.cpp .

$JAVAC -h . ../src/main/java/com/sri/yices/*.java

$CXX $LD_STATIC_FLAGS $CPPFLAGS $CXXFLAGS -c yicesJNI.cpp

$CXX $LD_STATIC_FLAGS $LDFLAGS -s -shared -o $YICES_2_JAVA_LIB_NAME yicesJNI.o $LIBS

cp $YICES_2_JAVA_LIB_NAME ../../
