package io.ksmt.solver.async

import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

interface KAsyncSolver<Config : KSolverConfiguration> : KSolver<Config> {

    suspend fun configureAsync(configurator: Config.() -> Unit)

    suspend fun assertAsync(expr: KExpr<KBoolSort>)

    suspend fun assertAsync(exprs: List<KExpr<KBoolSort>>) = exprs.forEach { assertAsync(it) }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>)

    suspend fun assertAndTrackAsync(exprs: List<KExpr<KBoolSort>>) = exprs.forEach { assertAndTrackAsync(it) }

    suspend fun pushAsync()

    suspend fun popAsync(n: UInt)

    suspend fun checkAsync(timeout: Duration): KSolverStatus

    suspend fun checkWithAssumptionsAsync(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus

    suspend fun modelAsync(): KModel

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>>

    suspend fun reasonOfUnknownAsync(): String

    suspend fun interruptAsync()
}
