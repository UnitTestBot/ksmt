---
layout: default
title: Advanced usage
nav_order: 3
---
# Advanced usage
{: .no_toc }

For basic KSMT usage, please refer to [Getting started](getting-started) guide.

Having tried the essential scenarios, find the 
[advanced example](https://github.com/UnitTestBot/ksmt/tree/main/examples/src/main/kotlin/AdvancedExamples.kt) and 
proceed to advanced usage.

---
<details open markdown="block">
  <summary>
    Table of contents:
  </summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

---
## Working with SMT formulas

Learn how to parse, simplify, and substitute expressions.

---
### Parsing formulas in SMT-LIB2 format

KSMT provides an API for parsing formulas in the SMT-LIB2 format.
Currently, KSMT provides a parser implemented on top of the Z3 solver API, and therefore `ksmt-z3` module is 
required for parsing.

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

---
### Default simplification

By default, `KContext` attempts to apply lightweight simplifications when you create an expression. If you do not 
need simplifications, disable them using the `KContext.simplificationMode` parameter.

```kotlin
// Simplification is enabled by default
val simplifyingContext = KContext()

// Disable simplifications on a context level
val nonSimplifyingContext = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)

val simplifiedExpr = with(simplifyingContext) {
   val a by boolSort
   !(a and falseExpr)
}

val nonSimplifiedExpr = with(nonSimplifyingContext) {
   val a by boolSort
   !(a and falseExpr)
}

println(nonSimplifiedExpr) // (not (and a false))
println(simplifiedExpr) // true
```

---
### Manual simplification

KSMT provides `KExprSimplifier`, so you can manually simplify an arbitrary expression.

```kotlin
// Context does not simplify expressions during creation
val ctx = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)

with(ctx) {
   val a by boolSort
   val nonSimplifiedExpr = !(a and falseExpr)

   val simplifier = KExprSimplifier(ctx)
   val simplifiedExpr = simplifier.apply(nonSimplifiedExpr)

   println(nonSimplifiedExpr) // (not (and a false))
   println(simplifiedExpr) // true
}
```

---
### Expression substitution

KSMT provides `KExprSubstitutor`, so you can replace all the expression occurrences with another
expression.

```kotlin
val a by boolSort
val b by boolSort
val expr = !(a and b)

val substitutor = KExprSubstitutor(this).apply {
   // Substitute all occurrences of `b` with `false`
   substitute(b, falseExpr)
}

val exprAfterSubstitution = substitutor.apply(expr)

println(expr) // (not (and a b))
println(exprAfterSubstitution) // true
```

---
## Working with SMT solvers

Learn how to configure and run solvers, get models, and switch to portfolio mode. 

---
### Solver configuration

KSMT provides an API for modifying solver-specific parameters.
Since the parameters and their correct values are solver-specific,
KSMT does not perform any checks.
See corresponding solver documentation for a list of available options.

```kotlin
with(ctx) {
   KZ3Solver(this).use { solver ->
      solver.configure {
         // set Z3 solver `random_seed` parameter to 42 
         setZ3Option("random_seed", 42)
      }
   }
}
```

---
### Solver-independent models

By default, SMT solver models are lazily initialized.
The values of the declared variables are loaded from the underlying solver native model on demand.
Therefore, models become invalid as soon as the solver closes. Moreover, solvers like Bitwuzla invalidate their models 
each time `check-sat` is called.
To overcome these problems, KSMT provides the `KModel.detach` function that allows you to make the model independent of
the underlying native representation.

```kotlin
val a by boolSort
val b by boolSort
val expr = a and b

val (model, detachedModel) = KZ3Solver(this).use { solver ->
   solver.assert(expr)
   println(solver.check()) // SAT
   val model = solver.model()

   // Detach model from solver
   val detachedModel = model.detach()

   model to detachedModel
}

try {
   model.eval(expr)
} catch (ex: Exception) {
   println("Model no longer valid after solver close")
}

println(detachedModel.eval(expr)) // true
```

Note: it is recommended to use `KModel.detach` when you need to keep the model in a `List`, for example.

---
### Solver runner

SMT solvers may ignore timeouts, or they suddenly crash, thus interrupting the entire application process.
KSMT provides a process-based solver runner: it runs each solver in a separate process.

```kotlin
// Create a long-lived solver manager that manages a pool of solver workers
KSolverRunnerManager().use { solverManager ->

   // Use solver API as usual
   with(ctx) {
      val a by boolSort
      val b by boolSort
      val expr = a and b

      // Create solver using manager instead of direct constructor invocation
      solverManager.createSolver(this, KZ3Solver::class).use { solver ->
         solver.assert(expr)
         println(solver.check()) // SAT
      }
   }
}
```

---
### Using custom solvers in a runner

Solver runner also supports user-defined solvers. Custom solvers must be registered via `registerSolver` before being used in the runner.

```kotlin
// Create a long-lived solver manager that manages a pool of solver workers
KSolverRunnerManager().use { solverManager ->
   // Register the user-defined solver in a current manager
   solverManager.registerSolver(CustomZ3BasedSolver::class, KZ3SolverUniversalConfiguration::class)

   // Use solver API as usual
   with(ctx) {
      val a by boolSort
      val b by boolSort
      val expr = a and b

      // Create solver using manager instead of direct constructor invocation
      solverManager.createSolver(this, CustomZ3BasedSolver::class).use { solver ->
         solver.assert(expr)
         println(solver.check()) // SAT
      }
   }
}
```

---
### Solver portfolio

To run solvers in portfolio mode, i.e., to run them in parallel until you get the first result, try the following 
workflow, which is similar to using the solver runner:

```kotlin
// Create a long-lived portfolio solver manager that manages a pool of solver workers
KPortfolioSolverManager(
   // Solvers to include in portfolio
   listOf(KZ3Solver::class, CustomZ3BasedSolver::class)
).use { solverManager ->
   // Since we use the user-defined solver in our portfolio, we must register it in the current manager
   solverManager.registerSolver(CustomZ3BasedSolver::class, KZ3SolverUniversalConfiguration::class)

   // Use solver API as usual
   with(ctx) {
      val a by boolSort
      val b by boolSort
      val expr = a and b

      // Create portfolio solver using manager
      solverManager.createPortfolioSolver(this).use { solver ->
         solver.assert(expr)
         println(solver.check()) // SAT
      }
   }
}
```
