# KSMT
Kotlin/Java API for various SMT solvers.

[![KSMT: build](https://github.com/UnitTestBot/ksmt/actions/workflows/build-and-run-tests.yml/badge.svg)](https://github.com/UnitTestBot/ksmt/workflows/build-and-run-tests.yml)
![Maven Central](https://img.shields.io/maven-central/v/io.ksmt/ksmt-core)
[![javadoc](https://javadoc.io/badge2/io.ksmt/ksmt-core/javadoc.svg)](https://javadoc.io/doc/io.ksmt/ksmt-core)

# Overview

`KSMT` provides a simple and unified way to work with SMT:
* Define formulas in a simple and solver-independent way
* Simplify and transform formulas without involving the SMT solver
* Solve SMT formulas using various SMT solvers through a unified API

# Getting started
Install via [Gradle](https://gradle.org/).

```kotlin
// core 
implementation("io.ksmt:ksmt-core:0.5.3")
// z3 solver
implementation("io.ksmt:ksmt-z3:0.5.3")
```

## Usage
Check out our [Getting started guide](docs/getting-started.md) and the [example project](examples).
Also, check out the [Java examples](examples/src/main/java).

# Features
Currently, KSMT supports the following SMT solvers:

| SMT Solver                                       |     Linux-x64      |    Windows-x64     |   MacOS-aarch64    |     MacOS-x64      |
|--------------------------------------------------|:------------------:|:------------------:|:------------------:|:------------------:|
| [Z3](https://github.com/Z3Prover/z3)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Bitwuzla](https://github.com/bitwuzla/bitwuzla) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [Yices2](https://github.com/SRI-CSL/yices2)      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [cvc5](https://github.com/cvc5/cvc5)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |

KSMT can express formulas in the following theories:

| Theory                  |         Z3         |      Bitwuzla      |       Yices2       |        cvc5        |
|-------------------------|:------------------:|:------------------:|:------------------:|:------------------:|
| Bitvectors              | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Arrays                  | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| IEEE Floats             | :heavy_check_mark: | :heavy_check_mark: |                    | :heavy_check_mark: |
| Uninterpreted Functions | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| Arithmetic              | :heavy_check_mark: |                    | :heavy_check_mark: | :heavy_check_mark: |

Check out our [roadmap](Requirements.md) for detailed description of features and future plans.

# Why KSMT?

### Kotlin based DSL for SMT formulas
KSMT provides simple and concise DSL for expressions.
```kotlin
val array by mkArraySort(intSort, intSort)
val index by intSort
val value by intSort

val expr = (array.select(index - 1.expr) lt value) and
        (array.select(index + 1.expr) gt value)
```
Check out our [example project](examples) for more complicated examples.

### Solver agnostic SMT formula representation
KSMT provides a solver-independent formula representation
with back and forth transformations to the solver's native representation.
Such representation allows introspection of formulas and transformation of formulas 
independent of the solver.

### Process based solver runner
KSMT provides a high-performant API for running SMT solvers in separate processes.
This feature is important for implementing hard timeouts and 
solving using multiple solvers in portfolio mode.

### Formula simplification and manipulation utils
KSMT provides a simplification engine that can simplify and especially evaluate all supported expressions in all
supported theories.
Also, KSMT provides utilities for performing formula transformation, visiting and expression substitution.

Check out our [advanced features tutorial](docs/advanced-usage.md) for the examples.
