package io.ksmt.solver.maxsmt.solvers

import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

interface KMaxSMTSolverInterface<C> : KSolver<C> where C : KSolverConfiguration {
    fun assertSoft(expr: KExpr<KBoolSort>, weight: UInt)

    fun checkMaxSMT(timeout: Duration = Duration.INFINITE, collectStatistics: Boolean = false): KMaxSMTResult

    fun collectMaxSMTStatistics(): KMaxSMTStatistics
}
