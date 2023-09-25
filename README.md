# KSMT

Meet the unified Kotlin/Java API for various SMT solvers.

Get the most out of SMT solving with KSMT features:
* Supporting more [solvers and theories](#supported-solvers-and-theories) — for all popular operating systems
* [Solver-agnostic formula representation](#solver-agnostic-formula-representation) and easy-to-use [DSL](#kotlin-based-dsl-for-smt-formulas)
* Utilities to [simplify and transform](#utilities-to-simplify-and-transform-expressions) your expressions
* Switching between solvers and support for [portfolio mode](#using-multiple-solvers--portfolio-mode-)
* Running solvers in a [separate process](#running-solvers-in-separate-processes) to reduce timeout-related crashes
* Streamlined [solver delivery](#ksmt-distribution) with no need for building a solver or implementing JVM bindings

[![KSMT: build](https://github.com/UnitTestBot/ksmt/actions/workflows/build-and-run-tests.yml/badge.svg)](https://github.com/UnitTestBot/ksmt/actions/workflows/build-and-run-tests.yml)
[![Maven Central](https://img.shields.io/maven-central/v/io.ksmt/ksmt-core)](https://central.sonatype.com/artifact/io.ksmt/ksmt-core/0.5.8)
[![javadoc](https://javadoc.io/badge2/io.ksmt/ksmt-core/javadoc.svg)](https://javadoc.io/doc/io.ksmt/ksmt-core)

## Get started

To start using KSMT, install it via [Gradle](https://gradle.org/):

```kotlin
// core 
implementation("io.ksmt:ksmt-core:0.5.8")
// z3 solver
implementation("io.ksmt:ksmt-z3:0.5.8")
```

Find basic instructions in the [Getting started](docs/getting-started.md) guide and try it out with the 
[Kotlin](examples/src/main/kotlin) or [Java](examples/src/main/java) examples.

To go beyond the basic scenarios, proceed to the [Advanced usage](docs/advanced-usage.md) guide and try the [advanced 
example](/examples/src/main/kotlin/AdvancedExamples.kt).

To get guided experience in KSMT, step through the detailed scenario for creating 
[custom expressions](docs/custom-expressions.md).

## Find more on KSMT features

Check the [Roadmap](https://github.com/UnitTestBot/ksmt/blob/main/Requirements.md) to know more about current
feature support and plans for the nearest future.

### Supported solvers and theories

KSMT provides support for various solvers:

| SMT solver                                       |     Linux-x64      |    Windows-x64     |   macOS-aarch64    |     macOS-x64      |
|--------------------------------------------------|:------------------:|:------------------:|:------------------:|:------------------:|
| [Z3](https://github.com/Z3Prover/z3)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Bitwuzla](https://github.com/bitwuzla/bitwuzla) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [Yices2](https://github.com/SRI-CSL/yices2)      | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |
| [cvc5](https://github.com/cvc5/cvc5)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |                    |

You can also use SMT solvers across multiple theories:

| Theory                  |         Z3         |      Bitwuzla      |         Yices2          |        cvc5        |
|-------------------------|:------------------:|:------------------:|:-----------------------:|:------------------:|
| Bitvectors              | :heavy_check_mark: | :heavy_check_mark: |   :heavy_check_mark:    | :heavy_check_mark: |
| Arrays                  | :heavy_check_mark: | :heavy_check_mark: |   :heavy_check_mark:    | :heavy_check_mark: |
| IEEE Floats             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: [^1] | :heavy_check_mark: |
| Uninterpreted Functions | :heavy_check_mark: | :heavy_check_mark: |   :heavy_check_mark:    | :heavy_check_mark: |
| Arithmetic              | :heavy_check_mark: |                    |   :heavy_check_mark:    | :heavy_check_mark: |

[^1]: IEEE Floats are supported in Yices2 using ksmt-symfpu 

### Solver-agnostic formula representation

Various scenarios are available for using SMT solvers: you can use a single solver to determine whether a formula is
satisfiable, or you can apply several solvers to the same expression successively. In the latter case, you need a _solver-agnostic formula representation_.

We implemented it in KSMT, so you can
* transform expressions from the solver native representation to KSMT representation and vice versa,
* use _formula introspection_,
* manipulate expressions without involving a solver,
* use expressions even if the solver is freed.

Expression interning (hash consing) affords faster expression comparison: we do not need to compare the expression
trees. Expressions are deduplicated, so we avoid redundant memory allocation.

### Kotlin-based DSL for SMT formulas

KSMT provides you with a unified DSL for SMT expressions:

```kotlin
val array by mkArraySort(intSort, intSort)
val index by intSort
val value by intSort

val expr = (array.select(index - 1.expr) lt value) and
        (array.select(index + 1.expr) gt value)
```

### Utilities to simplify and transform expressions

KSMT provides a simplification engine applicable to all supported expressions for all supported theories:

* reduce expression size and complexity
* evaluate expressions (including those with free variables) — reduce your expression to a constant
* perform formula transformations
* substitute expressions

KSMT simplification engine implements more than 200 rules.
By default, it attempts to apply simplifications when you create the expressions, but you can turn this
feature off, if necessary. You can also simplify an arbitrary expression manually.

### Using multiple solvers (portfolio mode)

SMT solvers may differ in performance across different formulas:
see the [International Satisfiability Modulo Theories Competition](https://smt-comp.github.io/2022/).

With KSMT portfolio solving, you can run multiple solvers in parallel on the same formula — until you get the first
(the fastest) result.

For detailed instructions on running multiple solvers, see [Advanced usage](docs/advanced-usage.md).

### Running solvers in separate processes

Most of the SMT solvers are research projects — they are implemented via native libraries and are sometimes not 
production ready:
* they may ignore timeouts — they sometimes hang in a long-running operation, and you cannot interrupt it;
* they may suddenly crash interrupting the entire process — because of a pointer issue, for example.

KSMT runs each solver in a separate process, which adds to stability of your application and provides support for
portfolio mode.

### KSMT distribution

Many solvers do not have prebuilt binaries, or binaries are for Linux only.

KSMT is distributed as JVM library with solver binaries provided. The library has been tested against the SMT-LIB 
benchmarks. Documentation and examples are also available.
