#!/bin/bash

git --no-pager diff <(nm cmake-build-debug/CMakeFiles/bitwuzla_jni.dir/bitwuzla_jni.cpp.o | grep -o "Java.*" | sort) <(cat io_ksmt_solver_bitwuzla_bindings_Native.h | grep -o "Java.*" | sort)
