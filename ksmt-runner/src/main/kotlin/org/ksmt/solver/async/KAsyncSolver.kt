package org.ksmt.solver.async

import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration

interface KAsyncSolver<Config : KSolverConfiguration> : KSolver<Config> {

    suspend fun configureAsync(configurator: Config.() -> Unit)

    suspend fun assertAsync(expr: KExpr<KBoolSort>)

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>)

    suspend fun pushAsync()

    suspend fun popAsync(n: UInt)

    suspend fun checkAsync(timeout: Duration): KSolverStatus

    suspend fun checkWithAssumptionsAsync(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus

    suspend fun modelAsync(): KModel

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>>

    suspend fun reasonOfUnknownAsync(): String

    suspend fun interruptAsync()
}
