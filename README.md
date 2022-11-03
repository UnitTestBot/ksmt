# KSMT
Kotlin API for various SMT solvers.

[![KSMT: build](https://github.com/UnitTestBot/ksmt/actions/workflows/build-and-run-tests.yml/badge.svg)](https://github.com/UnitTestBot/ksmt/workflows/build-and-run-tests.yml)
[![JitPack](https://jitpack.io/v/UnitTestBot/ksmt.svg)](https://jitpack.io/#UnitTestBot/ksmt)

# Getting started
Install via [JitPack](https://jitpack.io/) and [Gradle](https://gradle.org/).

```kotlin
// JitPack repository
repositories {
    maven { url = uri("https://jitpack.io") }
}

// core 
implementation("com.github.UnitTestBot.ksmt:ksmt-core:0.2.1")
// z3 solver
implementation("com.github.UnitTestBot.ksmt:ksmt-z3:0.2.1")
```

## Usage
Check out our [Getting started guide](docs/getting-started.md) and the [example project](examples).
Also, check out the [Java examples](examples/src/main/java).

# Features
Currently, KSMT supports the following SMT solvers:

| SMT Solver                                       |     Linux-x64      |    Windows-x64     |     MacOS-x64      |
|--------------------------------------------------|:------------------:|:------------------:|:------------------:|
| [Z3](https://github.com/Z3Prover/z3)             | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Bitwuzla](https://github.com/bitwuzla/bitwuzla) | :heavy_check_mark: | :heavy_check_mark: |                    |

KSMT can express formulas in the following theories:

| Theory                  |         Z3         |      Bitwuzla      |
|-------------------------|:------------------:|:------------------:|
| Bitvectors              | :heavy_check_mark: | :heavy_check_mark: |
| Arrays                  | :heavy_check_mark: | :heavy_check_mark: |
| IEEE Floats             | :heavy_check_mark: |                    |
| Uninterpreted Functions | :heavy_check_mark: | :heavy_check_mark: |
| Arithmetic              | :heavy_check_mark: |                    |
