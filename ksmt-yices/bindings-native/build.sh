YICES_VERSION="2.6.4"

echo "Found java: $JAVA_HOME"
echo "Found yices: $YICES_DIST | version: $YICES_VERSION"
echo "Build for $OS"
echo "Shared lib extension: $LIB_EXTENSION"

JAVAC="$JAVA_HOME/bin/javac"
CXX="g++"

echo "Java compiler $JAVAC"
echo "C++ compiler $CXX"

CPPFLAGS="-I $JAVA_HOME/include -I $JAVA_HOME/include/$OS -I $YICES_DIST/include"
CXXFLAGS="-fpermissive -g -fPIC -O3 -m64"

# note: libgcc is gcc specific option. Remove if not compiling with gcc
LD_STATIC_FLAGS="-static-libstdc++ -static-libgcc"
LD_FLAGS="-L $YICES_DIST/lib -l:libyices.$LIB_EXTENSION.$YICES_VERSION -l:libgmp.a"

YICES_2_JAVA_LIB_NAME="libyices2java.$LIB_EXTENSION"

rm -rf yices2_java_bindings
git clone https://github.com/SRI-CSL/yices2_java_bindings
cd yices2_java_bindings
git checkout d9858e5

git apply ../yices-bindings-patch.patch

rm -rf build
mkdir build
cd build

cp ../src/main/java/com/sri/yices/yicesJNI.cpp .

$JAVAC -h . ../src/main/java/com/sri/yices/*.java

$CXX $LD_STATIC_FLAGS $CPPFLAGS $CXXFLAGS -c yicesJNI.cpp

$CXX $LD_STATIC_FLAGS $LD_FLAGS -s -shared -o $YICES_2_JAVA_LIB_NAME yicesJNI.o

cp $YICES_2_JAVA_LIB_NAME ../../
