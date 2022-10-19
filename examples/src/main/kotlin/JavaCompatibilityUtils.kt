import org.ksmt.expr.KExpr
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort

object JavaCompatibilityUtils {
    @JvmStatic
    fun solverAssert(solver: KSolver, expr: KExpr<KBoolSort>) = solver.assert(expr)

    @JvmStatic
    fun check(solver: KSolver): KSolverStatus = solver.check()
}
