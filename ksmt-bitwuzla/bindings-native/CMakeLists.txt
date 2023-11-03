cmake_minimum_required (VERSION 3.8)

PROJECT(bitwuzla_jni)

FIND_PACKAGE(JNI REQUIRED)

SET(CXX_STANDARD 17)
SET(BITWUZLA_ROOT "${PROJECT_SOURCE_DIR}/bitwuzla")

# Path to bitwuzla.h, enums.h, option.h
SET(BITWUZLA_INCLUDE "${BITWUZLA_ROOT}/include")

# Path to bitwuzla library (libbitwuzla.so)
SET(BITWUZLA_LIB "${BITWUZLA_ROOT}/build/src/${CMAKE_SHARED_LIBRARY_PREFIX}bitwuzla${CMAKE_SHARED_LIBRARY_SUFFIX}")

ADD_LIBRARY (bitwuzla_jni SHARED bitwuzla_jni.cpp bitwuzla_extension.cpp)

message(${BITWUZLA_INCLUDE})

TARGET_INCLUDE_DIRECTORIES(bitwuzla_jni PRIVATE ${JNI_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(bitwuzla_jni PRIVATE "./include"})
TARGET_INCLUDE_DIRECTORIES(bitwuzla_jni PRIVATE ${BITWUZLA_INCLUDE})

SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# For Windows build, comment the first line, and uncomment the second one (link statically with libstdc++, ligbcc, lpthreads)
TARGET_LINK_LIBRARIES(bitwuzla_jni PRIVATE ${BITWUZLA_LIB})
#TARGET_LINK_LIBRARIES(bitwuzla_jni PRIVATE ${BITWUZLA_LIB} -static-libstdc++ -static-libgcc -Wl,-Bstatic -lstdc++ -lpthread -Wl,-Bdynamic)
