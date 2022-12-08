package org.ksmt.solver.runner

import org.ksmt.KContext
import org.ksmt.runner.models.generated.SolverType
import org.ksmt.solver.KSolver
import kotlin.reflect.KClass

private const val KSMT_SOLVER_PACKAGE = "org.ksmt.solver"
private const val Z3_SOLVER_CLASS_NAME = "$KSMT_SOLVER_PACKAGE.z3.KZ3Solver"
private const val BITWUZLA_SOLVER_CLASS_NAME = "$KSMT_SOLVER_PACKAGE.bitwuzla.KBitwuzlaSolver"

val KClass<out KSolver>.solverType: SolverType
    get() = when (qualifiedName) {
        Z3_SOLVER_CLASS_NAME -> SolverType.Z3
        BITWUZLA_SOLVER_CLASS_NAME -> SolverType.Bitwuzla
        else -> error("Unsupported solver: ${this.qualifiedName}")
    }

private val z3SolverConstructor: (KContext) -> KSolver by lazy {
    val cls = Class.forName(Z3_SOLVER_CLASS_NAME)
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver })
}

private val bitwuzlaSolverConstructor: (KContext) -> KSolver by lazy {
    val cls = Class.forName(BITWUZLA_SOLVER_CLASS_NAME)
    val ctor = cls.getConstructor(KContext::class.java)
    ({ ctx: KContext -> ctor.newInstance(ctx) as KSolver })
}

fun SolverType.createInstance(ctx: KContext): KSolver = when (this) {
    SolverType.Z3 -> z3SolverConstructor(ctx)
    SolverType.Bitwuzla -> bitwuzlaSolverConstructor(ctx)
}
