cmake_minimum_required (VERSION 3.8)

PROJECT(bitwuzla_jni)

FIND_PACKAGE(JNI REQUIRED)

SET(CXX_STANDARD 17)

# Path to bitwuzla.h
SET(BITWUZLA_INCLUDE "${PROJECT_SOURCE_DIR}/bitwuzla/include")

# Path to Bitwuzla sources root (required for extensions)
SET(BITWUZLA_SRC_ROOT "${PROJECT_SOURCE_DIR}/bitwuzla/src")

# Path to bitwuzla library (libbitwuzla.so) 
SET(BITWUZLA_LIB "${PROJECT_SOURCE_DIR}/bitwuzla/lib/${CMAKE_SHARED_LIBRARY_PREFIX}bitwuzla${CMAKE_SHARED_LIBRARY_SUFFIX}")

ADD_LIBRARY (bitwuzla_jni SHARED bitwuzla_jni.cpp bitwuzla_extension.c)

TARGET_INCLUDE_DIRECTORIES(bitwuzla_jni PRIVATE ${JNI_INCLUDE_DIRS})
TARGET_INCLUDE_DIRECTORIES(bitwuzla_jni PRIVATE ${BITWUZLA_INCLUDE})

SET(BITWUZLA_SRC_INCLUDE "${BITWUZLA_SRC_ROOT}")
TARGET_INCLUDE_DIRECTORIES(bitwuzla_jni PRIVATE ${BITWUZLA_SRC_INCLUDE})

SET(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

TARGET_LINK_LIBRARIES(bitwuzla_jni PRIVATE ${BITWUZLA_LIB} -static-libgcc -static-libstdc++)
