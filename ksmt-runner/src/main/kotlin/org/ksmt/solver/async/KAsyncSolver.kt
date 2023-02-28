package org.ksmt.solver.async

import kotlinx.coroutines.runBlocking
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration

interface KAsyncSolver<Config : KSolverConfiguration> : KSolver<Config> {

    override fun configure(configurator: Config.() -> Unit) = runBlocking {
        configureAsync(configurator)
    }

    suspend fun configureAsync(configurator: Config.() -> Unit)

    override fun assert(expr: KExpr<KBoolSort>) = runBlocking {
        assertAsync(expr)
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>)

    override fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) = runBlocking {
        assertAndTrackAsync(expr, trackVar)
    }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>)

    override fun push() = runBlocking {
        pushAsync()
    }

    suspend fun pushAsync()

    override fun pop(n: UInt) = runBlocking {
        popAsync(n)
    }

    suspend fun popAsync(n: UInt)

    override fun check(timeout: Duration): KSolverStatus = runBlocking {
        checkAsync(timeout)
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        runBlocking {
            checkWithAssumptionsAsync(assumptions, timeout)
        }

    suspend fun checkWithAssumptionsAsync(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus

    override fun model(): KModel = runBlocking {
        modelAsync()
    }

    suspend fun modelAsync(): KModel

    override fun unsatCore(): List<KExpr<KBoolSort>> = runBlocking {
        unsatCoreAsync()
    }

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>>

    override fun reasonOfUnknown(): String = runBlocking {
        reasonOfUnknownAsync()
    }

    suspend fun reasonOfUnknownAsync(): String

    override fun interrupt() = runBlocking {
        interruptAsync()
    }

    suspend fun interruptAsync()
}
