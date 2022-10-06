package org.ksmt.solver.runner

import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.KsmtWorkerBase
import org.ksmt.runner.generated.AssertParams
import org.ksmt.runner.generated.CheckParams
import org.ksmt.runner.generated.CheckWithAssumptionsParams
import org.ksmt.runner.generated.CreateSolverParams
import org.ksmt.runner.generated.PopParams
import org.ksmt.runner.generated.SolverProtocolModel
import org.ksmt.runner.generated.SolverType
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import kotlin.time.Duration

class KSolverRunner(
    private val hardTimeout: Duration,
    private val manager: KSolverRunnerManager,
    internal val worker: KsmtWorkerBase<SolverProtocolModel>,
) : KSolver, AutoCloseable {

    var active = true
        private set

    override fun close() {
        if (!active) return
        active = false
        runBlocking {
            deleteSolver()
        }
        manager.deleteSolver(this@KSolverRunner)
    }

    private fun terminate() {
        active = false
        manager.terminateSolver(this@KSolverRunner)
    }

    private fun ensureActive() {
        check(active) { "Solver is already closed" }
        if (!worker.isAlive) {
            throw KSolverException("Worker is not alive")
        }
    }

    override fun assert(expr: KExpr<KBoolSort>) = runBlocking {
        assertAsync(expr)
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) {
        ensureActive()
        val params = AssertParams(expr)
        withTimeoutAndExceptionHandling {
            worker.protocolModel.assert.startSuspending(worker.lifetime, params)
        }
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort> = runBlocking {
        assertAndTrackAsync(expr)
    }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>): KExpr<KBoolSort> {
        ensureActive()
        val params = AssertParams(expr)
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.assertAndTrack.startSuspending(worker.lifetime, params)
        }

        @Suppress("UNCHECKED_CAST")
        return result.expression as KExpr<KBoolSort>
    }

    override fun push(): Unit = runBlocking {
        pushAsync()
    }

    suspend fun pushAsync() {
        ensureActive()
        withTimeoutAndExceptionHandling {
            worker.protocolModel.push.startSuspending(worker.lifetime, Unit)
        }
    }

    override fun pop(n: UInt): Unit = runBlocking {
        popAsync(n)
    }

    suspend fun popAsync(n: UInt) {
        ensureActive()
        val params = PopParams(n)
        withTimeoutAndExceptionHandling {
            worker.protocolModel.pop.startSuspending(worker.lifetime, params)
        }
    }

    override fun check(timeout: Duration): KSolverStatus = runBlocking {
        checkAsync(timeout)
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus {
        ensureActive()
        val params = CheckParams(timeout.inWholeMilliseconds)
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.check.startSuspending(worker.lifetime, params)
        }
        return result.status
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = runBlocking {
        checkWithAssumptionsAsync(assumptions, timeout)
    }

    suspend fun checkWithAssumptionsAsync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus {
        ensureActive()
        val params = CheckWithAssumptionsParams(assumptions, timeout.inWholeMilliseconds)
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.checkWithAssumptions.startSuspending(worker.lifetime, params)
        }
        return result.status
    }

    override fun model(): KModel = runBlocking {
        modelAsync()
    }

    @Suppress("UNCHECKED_CAST")
    suspend fun modelAsync(): KModel {
        ensureActive()
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.model.startSuspending(worker.lifetime, Unit)
        }
        val interpretations = result.declarations.zip(result.interpretations) { decl, interp ->
            val interpEntries = interp.entries.map {
                KModel.KFuncInterpEntry(it.args as List<KExpr<*>>, it.value as KExpr<KSort>)
            }

            val functionInterp = KModel.KFuncInterp(
                interp.sort as KSort,
                interp.vars as List<KDecl<*>>,
                interpEntries,
                interp.default as? KExpr<KSort>?
            )
            (decl as KDecl<*>) to functionInterp
        }
        return KModelImpl(worker.astSerializationCtx.ctx, interpretations.toMap())
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = runBlocking {
        unsatCoreAsync()
    }

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> {
        ensureActive()
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.unsatCore.startSuspending(worker.lifetime, Unit)
        }

        @Suppress("UNCHECKED_CAST")
        return result.core as List<KExpr<KBoolSort>>
    }

    override fun reasonOfUnknown(): String = runBlocking {
        reasonOfUnknownAsync()
    }

    suspend fun reasonOfUnknownAsync(): String {
        ensureActive()
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.reasonOfUnknown.startSuspending(worker.lifetime, Unit)
        }
        return result.reasonUnknown
    }

    internal suspend fun initSolver(solverType: SolverType) {
        ensureActive()
        val params = CreateSolverParams(solverType)
        withTimeoutAndExceptionHandling {
            worker.protocolModel.initSolver.startSuspending(worker.lifetime, params)
        }
    }

    private suspend fun deleteSolver() {
        withTimeoutAndExceptionHandling {
            worker.protocolModel.deleteSolver.startSuspending(worker.lifetime, Unit)
        }
    }

    @Suppress("TooGenericExceptionCaught")
    private suspend inline fun <T> withTimeoutAndExceptionHandling(crossinline body: suspend () -> T): T {
        try {
            return withTimeout(hardTimeout) {
                body()
            }
        } catch (ex: RdFault) {
            throw KSolverException(ex)
        } catch (ex: Exception) {
            terminate()
            throw KSolverException(ex)
        }
    }
}
