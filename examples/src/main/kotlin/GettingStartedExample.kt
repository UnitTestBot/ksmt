import org.ksmt.KContext
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.utils.getValue
import kotlin.time.Duration.Companion.seconds

fun main() {
    val ctx = KContext()
    println("Basic usage example")
    basicSolverUsageExample(ctx)
    println("Push pop incremental example")
    pushPopIncrementalExample(ctx)
    println("Assumptions incremental example")
    assumptionsIncrementalExample(ctx)
    println("Unsat core generation example")
    unsatCoreGenerationExample(ctx)
}

private fun basicSolverUsageExample(ctx: KContext) =
    with(ctx) {
        // create symbolic variables
        val a by boolSort
        val b by intSort
        val c by intSort

        // create expression
        val constraint = a and (b ge c + 3.expr)

        KZ3Solver(this).use { solver -> // create s Z3 Smt solver instance
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

private fun pushPopIncrementalExample(ctx: KContext) =
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

private fun assumptionsIncrementalExample(ctx: KContext) =
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

private fun unsatCoreGenerationExample(ctx: KContext) =
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
