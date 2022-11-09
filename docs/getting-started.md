# Getting started

For code examples, please check out our [project](/examples).

## Installation

Installation via [JitPack](https://jitpack.io/) and [Gradle](https://gradle.org/).

#### 1. Enable JitPack repository in your build configuration:
```kotlin
// JitPack repository
repositories {
    maven { url = uri("https://jitpack.io") }
}
```

#### 2. Add KSMT core dependency:
```kotlin
dependencies {
    // core 
    implementation("com.github.UnitTestBot.ksmt:ksmt-core:0.2.1")    
}
```

#### 3. Add one or more SMT solver dependencies:
```kotlin
dependencies {
    // z3 
    implementation("com.github.UnitTestBot.ksmt:ksmt-z3:0.2.1")
    // bitwuzla
    implementation("com.github.UnitTestBot.ksmt:ksmt-bitwuzla:0.2.1")
}
```
SMT solver specific packages are provided with solver native binaries. 
Check OS compatibility [here](https://github.com/UnitTestBot/ksmt/tree/docs#features).

## KSMT usage

First, create a KSMT context that manages all expressions and solvers.
```kotlin
val ctx = KContext()
```

### Working with SMT formulas

Once the context is initialized, we can create expressions. 
In this example, we want to create an expression 

`a && (b >= c + 3)`

over Boolean variable `a` and integer variables `b` and `c`.

```kotlin
with(ctx) {
    // create symbolic variables
    val a by boolSort
    val b by intSort
    val c by intSort

    // create expression
    val constraint = a and (b ge c + 3.intExpr)
}
```
All KSMT expressions are typed and incorrect terms (e.g. `and` with integer arguments) 
result in a compile-time error.

### Working with SMT solvers

To check SMT formula satisfiability we need to instantiate an SMT solver. 
In this example, we use `constraint` from a previous step as an SMT formula 
and we use Z3 as an SMT solver.

```kotlin
with(ctx) {
    KZ3Solver(this).use { solver -> // create s Z3 SMT solver instance
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
The formula in the example above is satisfiable and we can obtain a model.
The model contains concrete values of our symbolic variables `a`, `b` and `c`
which evaluate our formula to `true`.

Note the use of kotlin `.use { }` construction 
which is useful for releasing all native resources acquired by the solver.

### Solver incremental API

KSMT solver API provides two approaches to incremental formula solving: push/pop
and assumptions. 

#### Incremental solving with push/pop

The push and pop operations in the solver allow us to work with assertions in the same way as with a stack.
The push operation saves the currently asserted expressions onto the stack,  
while the pop operation removes previously pushed assertions.

```kotlin
with(ctx) {
    // create symbolic variables
    val cond1 by boolSort
    val cond2 by boolSort
    val a by mkBv32Sort()
    val b by mkBv32Sort()
    val c by mkBv32Sort()
    val goal by mkBv32Sort()

    KZ3Solver(this).use { solver ->
        // a == 0
        solver.assert(a eq mkBv(value = 0))
        // goal == 2
        solver.assert(goal eq mkBv(value = 2))

        // push assertions stack
        solver.push()

        // a == goal
        solver.assert(a eq goal)

        /**
         * Formula is unsatisfiable because we have
         * a == 0 && goal == 2 && a == goal
         */
        val check0 = solver.check(timeout = 1.seconds)
        println("check0 = $check0") // UNSAT

        // pop assertions stack. a == goal is removed
        solver.pop()

        /**
         * Formula is satisfiable now because we have
         * a == 0 && goal == 2
         */
        val check1 = solver.check(timeout = 1.seconds)
        println("check1 = $check1") // SAT

        // b == if (cond1) a + 1 else a
        solver.assert(b eq mkIte(cond1, mkBvAddExpr(a, mkBv(value = 1)), a))

        // push assertions stack
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

        // pop assertions stack. b == goal is removed
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

Assumption mechanism allows us to assert an expression for a single check 
without actually adding it to the assertions.
The following example shows how to implement the previous example using assumptions
instead of push and pop operations.

```kotlin
with(ctx) {
    // create symbolic variables
    val cond1 by boolSort
    val cond2 by boolSort
    val a by mkBv32Sort()
    val b by mkBv32Sort()
    val c by mkBv32Sort()
    val goal by mkBv32Sort()

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

### Solver unsat cores

If the asserted SMT formula is unsatisfiable, we can extract the [unsat core](https://en.wikipedia.org/wiki/Unsatisfiable_core). 
The unsat core is a subset of inconsistent assertions and assumptions.

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
         * Track variable e2Track will appear in unsat core
         * */
        val e2Track = solver.assertAndTrack(e2)

        /**
         * Check satisfiability with e3 assumed.
         * Formula is unsatisfiable because e1 is inconsistent with e2 and e3
         * */
        val check = solver.checkWithAssumptions(assumptions = listOf(e3))
        println("check = $check")

        // retrieve unsat core
        val core = solver.unsatCore()
        println("unsat core = $core") // [track!fresh!0, (not c)]

        // simply asserted expression cannot be in unsat core
        println("e1 in core = ${e1 in core}") // false
        /**
         * An expression added with assertAndTrack cannot be in unsat core.
         * The corresponding track variable is used instead of the expression itself.
         */
        println("e2 in core = ${e2 in core}") // false
        println("e2Track in core = ${e2Track in core}") // true

        //the assumed expression appears in unsat core as is
        println("e3 in core = ${e3 in core}") // true
    }
}
```

### Parsing formulas in SMT-LIB2 format

KSMT provides an API for parsing formulas in the SMT-LIB2 format. 
Currently, KSMT provides a parser implemented on top of the Z3 solver API 
and therefore `ksmt-z3` module is required for parsing.

```kotlin
val formula = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (>= x y))
    (assert (>= y x))
"""
with(ctx) {
    val assertions = KZ3SMTLibParser().parse(formula)
}
```
