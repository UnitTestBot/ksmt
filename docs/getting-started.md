# Getting started

For your first steps with KSMT, try our [code examples](/examples).

To check OS compatibility, see [Supported solvers and theories](../README.md#supported-solvers-and-theories).

Find basic instructions to get use of KSMT:

<!-- TOC -->
* [Installation](#installation)
* [Usage](#usage)
    * [Working with SMT formulas](#working-with-smt-formulas)
    * [Working with SMT solvers](#working-with-smt-solvers)
    * [Incremental solving: API](#incremental-solving--api)
        * [Incremental solving with push/pop operations](#incremental-solving-with-pushpop-operations)
        * [Incremental solving with assumptions](#incremental-solving-with-assumptions)
    * [Solver unsatisfiable cores](#solver-unsatisfiable-cores)
<!-- TOC -->

## Installation

Install KSMT via [Gradle](https://gradle.org/).

1. Enable Maven Central repository in your build configuration:

```kotlin
repositories {
    mavenCentral()
}
```

2. Add KSMT core dependency:

```kotlin
dependencies {
    // core 
    implementation("io.ksmt:ksmt-core:0.5.11")    
}
```

3. Add one or more SMT solver dependencies:

```kotlin
dependencies {
    // z3 
    implementation("io.ksmt:ksmt-z3:0.5.11")
    // bitwuzla
    implementation("io.ksmt:ksmt-bitwuzla:0.5.11")
}
```

SMT solver specific packages are provided with solver native binaries.

## Usage

Create a KSMT context that manages expressions and solvers:

```kotlin
val ctx = KContext()
```

### Working with SMT formulas

Once the context is initialized, you can create expressions.

In the example below, we create an expression 

`a && (b >= c + 3)`

over Boolean variable `a` and integer variables `b` and `c`:

```kotlin
import io.ksmt.utils.getValue

with(ctx) {
    // create symbolic variables
    val a by boolSort
    val b by intSort
    val c by intSort

    // create an expression
    val constraint = a and (b ge c + 3.expr)
}
```

KSMT expressions are typed, and incorrect terms (e.g., `and` with integer arguments) 
result in a compile-time error.

Note: `import getValue` is required when using the `by` keyword.
Alternatively, use `mkConst(name, sort)`. 

### Working with SMT solvers

To check SMT formula satisfiability, we need to instantiate an SMT solver. 
In this example, we use `constraint` from the previous step as an SMT formula. We use Z3 as an SMT solver.

```kotlin
with(ctx) {
    KZ3Solver(this).use { solver -> // create a Z3 SMT solver instance
        // assert expression
        solver.assert(constraint)
        
        // check assertions satisfiability with timeout
        val satisfiability = solver.check(timeout = 1.seconds)
        println(satisfiability) // SAT
        
        // obtain model
        val model = solver.model()
        
        println("$a = ${model.eval(a)}") // a = true
        println("$b = ${model.eval(b)}") // b = 0
        println("$c = ${model.eval(c)}") // c = -3
    }
}
```

The formula in the example above is satisfiable, so we can get a model.
The model contains concrete values of the symbolic variables `a`, `b`, and `c`, which evaluate the formula to `true`.

Note: the Kotlin `.use { }` construction allows releasing the solver-consumed resources.

### Incremental solving: API

KSMT solver API provides two approaches to incremental formula solving: using _push/pop_ operations and using 
_assumptions_. 

#### Incremental solving with push/pop operations

_Push_ and _pop_ operations in the solver allow us to work with assertions as if we deal with a stack.
The _push_ operation puts the asserted expressions onto the stack, while the _pop_ operation removes the pushed 
assertions.

```kotlin
with(ctx) {
    // create symbolic variables
    val cond1 by boolSort
    val cond2 by boolSort
    val a by bv32Sort
    val b by bv32Sort
    val c by bv32Sort
    val goal by bv32Sort

    KZ3Solver(this).use { solver ->
        // a == 0
        solver.assert(a eq mkBv(value = 0))
        // goal == 2
        solver.assert(goal eq mkBv(value = 2))

        // push assertions onto stack
        solver.push()

        // a == goal
        solver.assert(a eq goal)

        /**
         * Formula is unsatisfiable because we have
         * a == 0 && goal == 2 && a == goal
         */
        val check0 = solver.check(timeout = 1.seconds)
        println("check0 = $check0") // UNSAT

        // pop assertions from stack; a == goal is removed
        solver.pop()

        /**
         * Formula is satisfiable now because we have
         * a == 0 && goal == 2
         */
        val check1 = solver.check(timeout = 1.seconds)
        println("check1 = $check1") // SAT

        // b == if (cond1) a + 1 else a
        solver.assert(b eq mkIte(cond1, mkBvAddExpr(a, mkBv(value = 1)), a))

        // push assertions onto stack
        solver.push()

        // b == goal
        solver.assert(b eq goal)

        /**
         * Formula is unsatisfiable because we have
         * a == 0 && goal == 2 
         *      && b == if (cond1) a + 1 else a 
         *      && goal == b
         * where all possible values for b are only 0 and 1
         */
        val check2 = solver.check(timeout = 1.seconds)
        println("check2 = $check2") // UNSAT

        // pop assertions from stack. b == goal is removed
        solver.pop()

        /**
         * Formula is satisfiable now because we have
         * a == 0 && goal == 2 
         *      && b == if (cond1) a + 1 else a
         */
        val check3 = solver.check(timeout = 1.seconds)
        println("check3 = $check3") // SAT

        // c == if (cond2) b + 1 else b
        solver.assert(c eq mkIte(cond2, mkBvAddExpr(b, mkBv(value = 1)), b))

        // push assertions stack
        solver.push()

        // c == goal
        solver.assert(c eq goal)

        /**
         * Formula is satisfiable because we have
         * a == 0 && goal == 2 
         *      && b == if (cond1) a + 1 else a
         *      && c == if (cond2) b + 1 else b 
         *      && goal == c
         * where all possible values for b are 0 and 1
         * and for c we have 0, 1 and 2
         */
        val check4 = solver.check(timeout = 1.seconds)
        println("check4 = $check4") // SAT
    }
}
```

#### Incremental solving with assumptions

Assumption mechanism allows us to assert an expression for a single check without actually adding it to assertions.
The following example shows how to implement the previous example using assumptions
instead of push and pop operations.

```kotlin
with(ctx) {
    // create symbolic variables
    val cond1 by boolSort
    val cond2 by boolSort
    val a by bv32Sort
    val b by bv32Sort
    val c by bv32Sort
    val goal by bv32Sort

    KZ3Solver(this).use { solver ->
        // a == 0
        solver.assert(a eq mkBv(value = 0))
        // goal == 2
        solver.assert(goal eq mkBv(value = 2))

        /**
         * Formula is unsatisfiable because we have
         * a == 0 && goal == 2 && a == goal
         * Expression a == goal is assumed for current check
         */
        val check0 = solver.checkWithAssumptions(
            assumptions = listOf(a eq goal),
            timeout = 1.seconds
        )
        println("check0 = $check0") // UNSAT

        /**
         * Formula is satisfiable because we have
         * a == 0 && goal == 2
         */
        val check1 = solver.check(timeout = 1.seconds)
        println("check1 = $check1") // SAT

        // b == if (cond1) a + 1 else a
        solver.assert(b eq mkIte(cond1, mkBvAddExpr(a, mkBv(value = 1)), a))

        /**
         * Formula is unsatisfiable because we have
         * a == 0 && goal == 2
         *      && b == if (cond1) a + 1 else a
         *      && goal == b
         * where all possible values for b are only 0 and 1
         * Expression goal == b is assumed for current check
         */
        val check2 = solver.checkWithAssumptions(
            assumptions = listOf(b eq goal),
            timeout = 1.seconds
        )
        println("check2 = $check2") // UNSAT

        /**
         * Formula is satisfiable now because we have
         * a == 0 && goal == 2
         *      && b == if (cond1) a + 1 else a
         */
        val check3 = solver.check(timeout = 1.seconds)
        println("check3 = $check3") // SAT

        // c == if (cond2) b + 1 else b
        solver.assert(c eq mkIte(cond2, mkBvAddExpr(b, mkBv(value = 1)), b))

        /**
         * Formula is satisfiable because we have
         * a == 0 && goal == 2
         *      && b == if (cond1) a + 1 else a
         *      && c == if (cond2) b + 1 else b
         *      && goal == c
         * where all possible values for b are 0 and 1
         * and for c we have 0, 1 and 2
         * Expression goal == c is assumed for current check
         */
        val check4 = solver.checkWithAssumptions(
            assumptions = listOf(c eq goal),
            timeout = 1.seconds
        )
        println("check4 = $check4") // SAT
    }
}
```

### Solver unsatisfiable cores

If the asserted SMT formula is unsatisfiable, we can extract the unsatisfiable core. 
The unsatisfiable core is a subset of inconsistent assertions and assumptions.

```kotlin
with(ctx) {
    // create symbolic variables
    val a by boolSort
    val b by boolSort
    val c by boolSort

    val e1 = (a and b) or c
    val e2 = !(a and b)
    val e3 = !c

    KZ3Solver(this).use { solver ->
        // simply assert e1
        solver.assert(e1)

        /**
         * Assert and track e2
         * e2 will appear in unsat core
         * */
        solver.assertAndTrack(e2)

        /**
         * Check satisfiability with e3 assumed.
         * Formula is unsatisfiable because e1 is inconsistent with e2 and e3
         * */
        val check = solver.checkWithAssumptions(assumptions = listOf(e3))
        println("check = $check")

        // retrieve unsat core
        val core = solver.unsatCore()
        println("unsat core = $core") // [(not (and b a)), (not c)]

        // simply asserted expression cannot be in unsat core
        println("e1 in core = ${e1 in core}") // false

        // an expression added with `assertAndTrack` appears in unsat core as is
        println("e2 in core = ${e2 in core}") // true

        // the assumed expression appears in unsat core as is
        println("e3 in core = ${e3 in core}") // true
    }
}
```
