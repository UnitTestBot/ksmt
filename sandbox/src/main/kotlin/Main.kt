import io.ksmt.KContext
import io.ksmt.solver.neurosmt.KNeuroSMTSolver
import io.ksmt.utils.getValue
import kotlin.time.Duration.Companion.seconds

fun main() {
    val ctx = KContext()

    with(ctx) {
        // create symbolic variables
        val a by boolSort
        val b by intSort
        val c by intSort

        // create an expression
        val constraint = a and (b ge c + 3.expr) and (b * 2.expr gt 8.expr) and !a

        KNeuroSMTSolver(this).use { solver -> // create a Stub SMT solver instance
            // assert expression
            solver.assert(constraint)

            // check assertions satisfiability with timeout
            val satisfiability = solver.check(timeout = 1.seconds)
            println(satisfiability) // SAT
        }
    }
}