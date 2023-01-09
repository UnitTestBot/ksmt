package org.ksmt.solver.runner

import com.jetbrains.rd.util.AtomicReference
import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeout
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.KsmtWorkerSession
import org.ksmt.runner.models.generated.AssertParams
import org.ksmt.runner.models.generated.CheckParams
import org.ksmt.runner.models.generated.CheckWithAssumptionsParams
import org.ksmt.runner.models.generated.CreateSolverParams
import org.ksmt.runner.models.generated.PopParams
import org.ksmt.runner.models.generated.SolverProtocolModel
import org.ksmt.runner.models.generated.SolverType
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import kotlin.time.Duration

class KSolverRunner<Config: KSolverConfiguration>(
    private val ctx: KContext,
    private val hardTimeout: Duration,
    private val worker: KsmtWorkerSession<SolverProtocolModel>,
    private val configurationBuilder: KSolverUniversalConfigurationBuilder<Config>,
) : KSolver<Config> {

    private val lastReasonOfUnknown = AtomicReference<String?>(null)

    override fun close() {
        runBlocking {
            suppressAllRunnerExceptions {
                deleteSolver()
            }
        }
        worker.release()
    }

    private fun terminate() {
        worker.terminate()
    }

    private fun ensureActive() {
        if (!worker.isAlive) {
            throw KSolverException("Solver worker is terminated")
        }
    }

    override fun configure(configurator: Config.() -> Unit) = runBlocking {
        configureAsync(configurator)
    }

    suspend fun configureAsync(configurator: Config.() -> Unit) {
        ensureActive()
        val config = configurationBuilder.build { configurator() }
        withTimeoutAndExceptionHandling {
            worker.protocolModel.configure.startSuspending(worker.lifetime, config)
        }
    }

    override fun assert(expr: KExpr<KBoolSort>) = runBlocking {
        assertAsync(expr)
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) {
        ctx.ensureContextMatch(expr)
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
        ctx.ensureContextMatch(expr)
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
        return handleCheckTimeoutAsUnknown {
            val result = withTimeoutAndExceptionHandling {
                worker.protocolModel.check.startSuspending(worker.lifetime, params)
            }
            result.status
        }
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
        ctx.ensureContextMatch(assumptions)
        ensureActive()

        val params = CheckWithAssumptionsParams(assumptions, timeout.inWholeMilliseconds)
        return handleCheckTimeoutAsUnknown {
            val result = withTimeoutAndExceptionHandling {
                worker.protocolModel.checkWithAssumptions.startSuspending(worker.lifetime, params)
            }
            result.status
        }
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
                interp.decl as KDecl<KSort>,
                interp.vars as List<KDecl<*>>,
                interpEntries,
                interp.default as? KExpr<KSort>?
            )
            (decl as KDecl<*>) to functionInterp
        }
        val uninterpretedSortUniverse = result.uninterpretedSortUniverse.associateBy(
            { entry -> entry.sort as KUninterpretedSort },
            { entry -> entry.universe.mapTo(hashSetOf()) { it as KExpr<KUninterpretedSort> } }
        )
        return KModelImpl(worker.astSerializationCtx.ctx, interpretations.toMap(), uninterpretedSortUniverse)
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

    suspend fun reasonOfUnknownAsync(): String = lastReasonOfUnknown.updateIfNull {
        ensureActive()
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.reasonOfUnknown.startSuspending(worker.lifetime, Unit)
        }
        result.reasonUnknown
    }

    override fun interrupt() = runBlocking {
        interruptAsync()
    }

    suspend fun interruptAsync() {
        ensureActive()
        withTimeoutAndExceptionHandling {
            worker.protocolModel.interrupt.startSuspending(worker.lifetime, Unit)
        }
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

    @Suppress("TooGenericExceptionCaught")
    private suspend inline fun suppressAllRunnerExceptions(crossinline body: suspend () -> Unit) {
        try {
            body()
        } catch (ex: Exception) {
            // Propagate exceptions caused by the exceptions on remote side.
            if (ex is KSolverException && ex.cause is RdFault) {
                throw ex
            }
        }
    }

    private suspend inline fun handleCheckTimeoutAsUnknown(
        crossinline body: suspend () -> KSolverStatus
    ): KSolverStatus {
        try {
            lastReasonOfUnknown.getAndSet(null)
            return body()
        } catch (ex: KSolverException) {
            val cause = ex.cause
            if (cause is TimeoutCancellationException) {
                lastReasonOfUnknown.getAndSet("timeout: ${cause.message}")
                return KSolverStatus.UNKNOWN
            }
            throw ex
        }
    }

    private suspend inline fun <T> AtomicReference<T?>.updateIfNull(
        crossinline body: suspend () -> T
    ): T {
        val oldValue = get()
        if (oldValue != null) return oldValue

        val newValue = body()

        getAndSet(newValue)
        return newValue
    }
}
