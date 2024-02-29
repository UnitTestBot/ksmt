#!/bin/bash

JAVA_8_HOME="/usr/lib/jvm/java-8-temurin"
BITWUZLA_SRC_ROOT="/home/sobol/CLionProjects/bitwuzla"
BINDINGS_PACKAGE="io.ksmt.solver.bitwuzla.bindings"
bindings_classes_dir="../ksmt-bitwuzla-core/build/classes/kotlin/main"

bindings_fqn="${BINDINGS_PACKAGE}.Native"
current_dir="$PWD"

cd $bindings_classes_dir

${JAVA_8_HOME}/bin/javah ${bindings_fqn}

cp ./*.h "${current_dir}"
