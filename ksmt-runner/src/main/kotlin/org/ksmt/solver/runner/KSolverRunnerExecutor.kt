package org.ksmt.solver.runner

import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.runner.core.KsmtWorkerSession
import org.ksmt.runner.generated.models.AssertAndTrackParams
import org.ksmt.runner.generated.models.AssertParams
import org.ksmt.runner.generated.models.CheckParams
import org.ksmt.runner.generated.models.CheckWithAssumptionsParams
import org.ksmt.runner.generated.models.CreateSolverParams
import org.ksmt.runner.generated.models.PopParams
import org.ksmt.runner.generated.models.SolverConfigurationParam
import org.ksmt.runner.generated.models.SolverProtocolModel
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.solver.KSolverUnsupportedParameterException
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import kotlin.time.Duration

class KSolverRunnerExecutor(
    private val hardTimeout: Duration,
    private val worker: KsmtWorkerSession<SolverProtocolModel>,
) {

    suspend fun configureAsync(config: List<SolverConfigurationParam>) {
        ensureActive()

        queryWithTimeoutAndExceptionHandling {
            configure.startSuspending(worker.lifetime, config)
        }
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) {
        ensureActive()

        val params = AssertParams(expr)
        queryWithTimeoutAndExceptionHandling {
            assert.startSuspending(worker.lifetime, params)
        }
    }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        ensureActive()

        val params = AssertAndTrackParams(expr, trackVar)
        queryWithTimeoutAndExceptionHandling {
            assertAndTrack.startSuspending(worker.lifetime, params)
        }
    }

    suspend fun pushAsync() {
        ensureActive()

        queryWithTimeoutAndExceptionHandling {
            push.startSuspending(worker.lifetime, Unit)
        }
    }

    suspend fun popAsync(n: UInt) {
        ensureActive()

        val params = PopParams(n)
        queryWithTimeoutAndExceptionHandling {
            pop.startSuspending(worker.lifetime, params)
        }
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus {
        ensureActive()

        val params = CheckParams(timeout.inWholeMilliseconds)
        val result = queryWithTimeoutAndExceptionHandling {
            runCheckSatQuery {
                check.startSuspending(worker.lifetime, params)
            }
        }
        return result.status
    }

    suspend fun checkWithAssumptionsAsync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus {
        ensureActive()

        val params = CheckWithAssumptionsParams(assumptions, timeout.inWholeMilliseconds)
        val result = queryWithTimeoutAndExceptionHandling {
            runCheckSatQuery {
                checkWithAssumptions.startSuspending(worker.lifetime, params)
            }
        }
        return result.status
    }

    @Suppress("UNCHECKED_CAST")
    suspend fun modelAsync(): KModel {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandling {
            model.startSuspending(worker.lifetime, Unit)
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
            { entry -> entry.universe.mapTo(hashSetOf()) { it as KUninterpretedSortValue } }
        )
        return KModelImpl(worker.astSerializationCtx.ctx, interpretations.toMap(), uninterpretedSortUniverse)
    }

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandling {
            unsatCore.startSuspending(worker.lifetime, Unit)
        }

        @Suppress("UNCHECKED_CAST")
        return result.core as List<KExpr<KBoolSort>>
    }

    suspend fun reasonOfUnknownAsync(): String {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandling {
            reasonOfUnknown.startSuspending(worker.lifetime, Unit)
        }
        return result.reasonUnknown
    }

    suspend fun interruptAsync() {
        ensureActive()

        // No queries to interrupt
        if (!hasOngoingCheckSatQueries) {
            return
        }

        queryWithTimeoutAndExceptionHandling {
            interrupt.startSuspending(worker.lifetime, Unit)
        }
    }

    internal suspend fun initSolver(solverType: SolverType) {
        ensureActive()

        val params = CreateSolverParams(solverType)
        queryWithTimeoutAndExceptionHandling {
            initSolver.startSuspending(worker.lifetime, params)
        }
    }

    internal suspend fun deleteSolver() {
        ensureActive()

        queryWithTimeoutAndExceptionHandling {
            deleteSolver.startSuspending(worker.lifetime, Unit)
        }
        worker.release()
    }

    internal fun terminate() {
        worker.terminate()
    }

    internal fun terminateIfBusy() {
        if (hasOngoingCheckSatQueries) {
            terminate()
        }
    }

    private val ongoingCheckSatQueries = AtomicInteger(0)

    private val hasOngoingCheckSatQueries: Boolean
        get() = ongoingCheckSatQueries.get() != 0

    private suspend inline fun <T> runCheckSatQuery(
        crossinline body: suspend () -> T
    ): T = try {
        ongoingCheckSatQueries.incrementAndGet()
        body()
    } finally {
        ongoingCheckSatQueries.decrementAndGet()
    }

    private fun ensureActive() {
        if (!worker.isAlive) {
            throw KSolverExecutorNotAliveException()
        }
    }

    @Suppress(
        "TooGenericExceptionCaught",
        "SwallowedException",
        "ThrowsCount"
    )
    private suspend inline fun <T> queryWithTimeoutAndExceptionHandling(
        crossinline body: suspend SolverProtocolModel.() -> T
    ): T {
        try {
            return withTimeout(hardTimeout) {
                worker.protocolModel.body()
            }
        } catch (ex: RdFault) {
            throwSolverException(ex)
        } catch (ex: Throwable) {
            terminate()
            if (ex is TimeoutCancellationException) {
                throw KSolverExecutorTimeoutException(ex.message)
            } else {
                throw KSolverExecutorOtherException(ex)
            }
        }
    }

    private fun throwSolverException(reason: RdFault): Nothing = when (reason.reasonTypeFqn) {
        KSolverException::class.simpleName ->
            throw KSolverException(reason.reasonMessage)

        KSolverUnsupportedFeatureException::class.simpleName ->
            throw KSolverUnsupportedFeatureException(reason.reasonMessage)

        KSolverUnsupportedParameterException::class.simpleName ->
            throw KSolverUnsupportedParameterException(reason.reasonMessage)

        else -> throw KSolverException(reason)
    }
}
