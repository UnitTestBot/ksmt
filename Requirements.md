| Feature                                                                           | Status         |
|-----------------------------------------------------------------------------------|----------------|
| [Rich expression system](#rich-expression-system)                                 | Done           |
| [Expression interning](#expression-interning)                                     | Done           |
| [Basic theories support](#basic-theories-support)                                 | Done           |
| [SMT-LIB2 parser](#smt-lib2-parser)                                               | Done partially |
| [SMT-LIB2 serializer](#smt-lib2-serializer)                                       | TODO           |
| [Unsupported features handling](#unsupported-features-handling)                   | Done partially |
| [SMT solver support](#smt-solver-support)                                         | Done           |
| [Z3 solver support](#z3-solver-support)                                           | Done           |
| [Bitwuzla solver support](#bitwuzla-solver-support)                               | Done           |
| [Yices2 solver support](#yices2-solver-support)                                   | In progress    |
| [CVC5 solver support](#cvc5-solver-support)                                       | TODO           |
| [External process runner](#external-process-runner)                               | Done           |
| [Portfolio solver](#portfolio-solver)                                             | TODO           |
| [Solver configuration API](#solver-configuration-api)                             | In progress    |
| [Deployment](#deployment)                                                         | Done partially |
| [Expression simplification / evaluation](#expression-simplification-/-evaluation) | TODO           |
| ...                                                                               | -              |

### Rich expression system

Provide a solver-independent formula representation
with back and forth transformations to the solver's native representation.
Such representation allows introspection of formulas and transformation of formulas
independent of the solver. Check out [KASt](ksmt-core/src/main/kotlin/org/ksmt/KAst.kt) and its inheritors
for implementation details.

### Expression interning

Interning (aka [hash consing](https://en.wikipedia.org/wiki/Hash_consing)) is needed for:

1. **Constant time ast comparison.** Otherwise, we need to compare trees.
2. **Ast deduplication.** We can have many duplicate nodes (e.g. constants).

Currently, interning is implemented via [KContext](ksmt-core/src/main/kotlin/org/ksmt/KContext.kt) which
manages all created asts.

### Basic theories support

Support theories in KSMT expressions system. Each theory consists of:

1. **Expressions.** Theory specific operations over terms.
   All implementations are in [expr](ksmt-core/src/main/kotlin/org/ksmt/expr) package.
2. **Sorts.** Theory specific types.
   All implementations are in [sort](ksmt-core/src/main/kotlin/org/ksmt/sort) package.
3. **Declarations.** Declarations (name, argument sorts, result sort) of theory specific functions.
   All implementations are in [decl](ksmt-core/src/main/kotlin/org/ksmt/decl) package.

KSMT expression system supports following theories and their combinations:

1. [**Core**](https://smtlib.cs.uiowa.edu/theories-Core.shtml). Basic boolean operations.
2. [**BV**](https://smtlib.cs.uiowa.edu/theories-FixedSizeBitVectors.shtml). Bit vectors with arbitrary size.
3. **Arithmetic**. [Integers](https://smtlib.cs.uiowa.edu/theories-Ints.shtml),
   [Reals](https://smtlib.cs.uiowa.edu/theories-Reals.shtml),
   and their [combinations](https://smtlib.cs.uiowa.edu/theories-Reals_Ints.shtml).
4. [**FP**](https://smtlib.cs.uiowa.edu/theories-FloatingPoint.shtml). IEEE Floating point numbers.
5. [**Arrays**](https://smtlib.cs.uiowa.edu/theories-ArraysEx.shtml).
6. [**UF**](https://smtlib.cs.uiowa.edu/logics-all.shtml#QF_UF). Uninterpreted functions and sorts.
7. **Quantifiers**. Existential and universal quantifiers.

### SMT-LIB2 parser

Provide a parser for formulas, written in the SMT-LIB2 language.
Main goals are:

1. Allow the user to instantiate KSMT expressions from SMT-LIB.
2. Provide us the opportunity to use a rich database of [benchmarks](https://smtlib.cs.uiowa.edu/benchmarks.shtml).

Currently, [implemented](ksmt-z3/src/main/kotlin/org/ksmt/solver/z3/KZ3SMTLibParser.kt) on top of Z3 SMT solver.
A solver-agnostic implementation may be done in the future.

### SMT-LIB2 serializer

Provide a serializer for KSMT expressions in SMT-LIB2 format.

Motivation:

1. SMT-LIB is an easy way to interact with new SMT solver.
2. SMT-LIB format is well known to the community and is the most suitable way to visualize formulas.

Can be implemented on top of the Z3 SMT solver (easy version) or in a solver-independent way.

### Unsupported features handling

If some solver doesn't support some theory (e.g. BV) or some feature (e.g. unsat-core generation) we need to throw
specialized exception.
Currently, [KSolverUnsupportedFeatureException](ksmt-core/src/main/kotlin/org/ksmt/solver/KSolverUnsupportedFeatureException.kt)
is thrown.

To simplify the user experience, the exception may contain a recommendation to switch to another solver.
This recommendation feature may be implemented in the future.

### SMT solver support

Provide an interface to interact with various SMT solvers.
Features:

1. Expression assertion.
2. Check-sat with timeout.
3. Model generation.
4. [Unsat-core](https://en.wikipedia.org/wiki/Unsatisfiable_core) generation.
5. Incremental solving via push/pop.
6. Incremental solving via assumptions.

For implementation details, check out [KSolver](ksmt-core/src/main/kotlin/org/ksmt/solver/KSolver.kt).

### Z3 solver support

[Z3](https://github.com/Z3Prover/z3) is a well known production ready SMT solver.

Z3 has native support for all theories,
listed in [KSMT supported theories](#basic-theories-support)
and provides all the functionality, listed in [SMT solver features](#smt-solver-support).

For implementation details, check out [KZ3Solver](ksmt-z3/src/main/kotlin/org/ksmt/solver/z3/KZ3Solver.kt).

### Bitwuzla solver support

[Bitwuzla](https://github.com/bitwuzla/bitwuzla) is a research solver
based on [Boolector](https://github.com/Boolector/boolector).
Bitwuzla specializes on BV and Fp theories
and [performs well](https://smt-comp.github.io/2022/results/qf-bitvec-single-query)
on SMT-COMP.

Bitwuzla has native support for the following theories:

1. **BV**. Full support
2. **FP**. Full support
3. **Arrays**. [QF_ABVFP](https://smtlib.cs.uiowa.edu/logics-all.shtml#QF_ABV) only.
   No nested arrays allowed.

Other theories and nested arrays will not be supported.

For the solver features, listed in [SMT solver features](#smt-solver-support),
Bitwuzla provides full native support.

For implementation details, check
out [KBitwuzlaSolver](ksmt-bitwuzla/src/main/kotlin/org/ksmt/solver/bitwuzla/KBitwuzlaSolver.kt).

### Yices2 solver support

[Yices2](https://github.com/SRI-CSL/yices2) is
a [well performing](https://smt-comp.github.io/2022/results/qf-bitvec-single-query) SMT solver.

Yices2 has native support for all theories,
listed in [KSMT supported theories](#basic-theories-support), except FP.

Yices2 provides all the functionality, listed in [SMT solver features](#smt-solver-support).

For FP support [SymFpu](https://github.com/martin-cs/symfpu) approach can be used
(rewrite all FP expressions over BV terms).

### CVC5 solver support

[CVC5](https://github.com/cvc5/cvc5) is
a [well performing](https://smt-comp.github.io/2022/results/qf-bitvec-single-query) SMT solver.

CVC5 has native support for all theories,
listed in [KSMT supported theories](#basic-theories-support)
and provides all the functionality, listed in [SMT solver features](#smt-solver-support).

### External process runner

Run solvers in separate processes to preserve user applications stability.

SMT solvers are implemented via native libraries and usually have the following issues:

1. Timeout ignore. SMT solver may hang in a long-running operation before reaching a checkpoint.
   Since solver is a native library, there is no way to interrupt it.
2. Solvers are usually research projects with a bunch of bugs, e.g. pointer issues. Such
   errors lead to the interruption of the entire process, including the user's app.

Currently, we have a [process runner implementation](ksmt-runner), that allows us to
force the solver to respect timeouts and survive after native errors.

### Portfolio solver

Various SMT solvers usually show different performance on a same SMT formula.
[Portfolio solving](https://arxiv.org/abs/1505.03340) is a simple idea of
running different solvers on the same formula simultaneously and waiting only
for the first result.

This approach can be implemented on top of [external process runner](#external-process-runner).

### Solver configuration API

Extend SMT solver interface with an option to
pass solver specific parameters to the underlying solver.

For implementation details, check out [corresponding PR](https://github.com/UnitTestBot/ksmt/pull/34).


### Deployment

Deliver KSMT to end users. 

Currently, we use [JitPack](https://jitpack.io) for deployment.

Use Maven Central in the future.

### Expression simplification / evaluation

Implement simplification rules for KSMT expressions, and apply it to:
1. Expression simplification
2. Lightweight eager simplification during expressions creation
3. Expression evaluation wrt model (special case of simplification)




### Performance tests

Describe how to write performance tests that measure overhead of KSMT for typical operations.
After done, add results here

### Solver Native API support

Don't use Z3Java API

