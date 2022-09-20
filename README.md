# KSMT
Kotlin API for various SMT solvers

[![KSMT: build](https://github.com/UnitTestBot/ksmt/actions/workflows/build-and-run-tests.yml/badge.svg)](https://github.com/UnitTestBot/UTBotJava/ksmt/workflows/build-and-run-tests.yml)

# Getting started
Install via JitPack and Gradle

```kotlin
// core 
implementation("com.github.UnitTestBot.ksmt:ksmt-core:0.1.0")
// z3 solver
implementation("com.github.UnitTestBot.ksmt:ksmt-z3:0.1.0")
```

## Usage
See [examples](examples)

# Features
Currently KSMT supports the following SMT solvers:

| SMT Solver | Linux-x64 | Windows-x64 | MacOS-x64 |
| --- |:---:|:---:|:---:|
| [Z3](https://github.com/Z3Prover/z3) | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| [Bitwuzla](https://github.com/bitwuzla/bitwuzla) | :heavy_check_mark: | :heavy_check_mark: |  |

KSMT can express formulas in the following theories:
| Theory | Z3 | Bitwuzla |
| --- |:---:|:---:|
| Bitvectors | :heavy_check_mark: | :heavy_check_mark: |
| Arrays | :heavy_check_mark: | :heavy_check_mark: |
| IEEE Floats | :heavy_check_mark: |  |
| Uninterpreted Functions | :heavy_check_mark: | :heavy_check_mark: |
| Arithmetic | :heavy_check_mark: | |
