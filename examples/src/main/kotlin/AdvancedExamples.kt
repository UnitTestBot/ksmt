import io.ksmt.KContext
import io.ksmt.expr.rewrite.KExprSubstitutor
import io.ksmt.expr.rewrite.simplify.KExprSimplifier
import io.ksmt.solver.KSolver
import io.ksmt.solver.portfolio.KPortfolioSolverManager
import io.ksmt.solver.runner.KSolverRunnerManager
import io.ksmt.solver.z3.KZ3SMTLibParser
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import io.ksmt.solver.z3.KZ3SolverUniversalConfiguration
import io.ksmt.utils.getValue

fun main() {
    val ctx = KContext()

    parseSmtFormula(ctx)
    simplificationOnCreation()
    manualExpressionSimplification()
    expressionSubstitution(ctx)

    solverConfiguration(ctx)
    solverModelDetach(ctx)
    solverRunnerUsageExample(ctx)
    solverRunnerWithCustomSolverExample(ctx)
    solverPortfolioExample(ctx)
}

fun simplificationOnCreation() {
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
}

fun manualExpressionSimplification() {
    // Context do not simplify expressions during creation
    val ctx = KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)

    with(ctx) {
        val a by boolSort
        val nonSimplifiedExpr = !(a and falseExpr)

        val simplifier = KExprSimplifier(ctx)
        val simplifiedExpr = simplifier.apply(nonSimplifiedExpr)

        println(nonSimplifiedExpr) // (not (and a false))
        println(simplifiedExpr) // true
    }
}

fun expressionSubstitution(ctx: KContext) = with(ctx) {
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
}

fun parseSmtFormula(ctx: KContext) = with(ctx) {
    val formula = """
    (declare-fun x () Int)
    (declare-fun y () Int)
    (assert (>= x y))
    (assert (>= y x))
    """
    val assertions = KZ3SMTLibParser(this).parse(formula)
    println(assertions)
}

fun solverConfiguration(ctx: KContext) = with(ctx) {
    KZ3Solver(this).use { solver ->
        solver.configure {
            // set Z3 solver parameter random_seed to 42
            setZ3Option("random_seed", 42)
        }
    }
}

fun solverModelDetach(ctx: KContext) = with(ctx) {
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
}

fun solverRunnerUsageExample(ctx: KContext) {
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
}

// User defined solver
class CustomZ3BasedSolver(ctx: KContext) : KSolver<KZ3SolverConfiguration> by KZ3Solver(ctx) {
    init {
        configure {
            setZ3Option("smt.logic", "QF_BV")
        }
    }
}

fun solverRunnerWithCustomSolverExample(ctx: KContext) {
    // Create a long-lived solver manager that manages a pool of solver workers
    KSolverRunnerManager().use { solverManager ->
        // Register user-defined solver in a current manager
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
}

fun solverPortfolioExample(ctx: KContext) {
    // Create a long-lived portfolio solver manager that manages a pool of solver workers
    KPortfolioSolverManager(
        // Solvers to include in portfolio
        listOf(KZ3Solver::class, CustomZ3BasedSolver::class)
    ).use { solverManager ->
        // Since we use user-defined solver in our portfolio we must register it in the current manager
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
}
