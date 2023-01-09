package org.ksmt.solver.runner

import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.KsmtWorkerSession
import org.ksmt.runner.models.generated.AssertAndTrackParams
import org.ksmt.runner.models.generated.AssertParams
import org.ksmt.runner.models.generated.CheckParams
import org.ksmt.runner.models.generated.CheckWithAssumptionsParams
import org.ksmt.runner.models.generated.CreateSolverParams
import org.ksmt.runner.models.generated.PopParams
import org.ksmt.runner.models.generated.SolverConfigurationParam
import org.ksmt.runner.models.generated.SolverProtocolModel
import org.ksmt.runner.models.generated.SolverType
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import kotlin.time.Duration

class KSolverRunnerExecutor(
    private val hardTimeout: Duration,
    private val worker: KsmtWorkerSession<SolverProtocolModel>,
) {

    private fun ensureActive() {
        if (!worker.isAlive) {
            throw KSolverExecutorNotAliveException()
        }
    }

    suspend fun configureAsync(config: List<SolverConfigurationParam>) {
        ensureActive()

        withTimeoutAndExceptionHandling {
            worker.protocolModel.configure.startSuspending(worker.lifetime, config)
        }
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) {
        ensureActive()

        val params = AssertParams(expr)
        withTimeoutAndExceptionHandling {
            worker.protocolModel.assert.startSuspending(worker.lifetime, params)
        }
    }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        ensureActive()

        val params = AssertAndTrackParams(expr, trackVar)
        withTimeoutAndExceptionHandling {
            worker.protocolModel.assertAndTrack.startSuspending(worker.lifetime, params)
        }
    }

    suspend fun pushAsync() {
        ensureActive()

        withTimeoutAndExceptionHandling {
            worker.protocolModel.push.startSuspending(worker.lifetime, Unit)
        }
    }

    suspend fun popAsync(n: UInt) {
        ensureActive()

        val params = PopParams(n)
        withTimeoutAndExceptionHandling {
            worker.protocolModel.pop.startSuspending(worker.lifetime, params)
        }
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus {
        ensureActive()

        val params = CheckParams(timeout.inWholeMilliseconds)
        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.check.startSuspending(worker.lifetime, params)
        }
        return result.status
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

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> {
        ensureActive()

        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.unsatCore.startSuspending(worker.lifetime, Unit)
        }

        @Suppress("UNCHECKED_CAST")
        return result.core as List<KExpr<KBoolSort>>
    }

    suspend fun reasonOfUnknownAsync(): String {
        ensureActive()

        val result = withTimeoutAndExceptionHandling {
            worker.protocolModel.reasonOfUnknown.startSuspending(worker.lifetime, Unit)
        }
        return result.reasonUnknown
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

    internal suspend fun deleteSolver() {
        ensureActive()

        withTimeoutAndExceptionHandling {
            worker.protocolModel.deleteSolver.startSuspending(worker.lifetime, Unit)
        }
        worker.release()
    }

    internal fun terminate() {
        worker.terminate()
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
            if (ex is TimeoutCancellationException) {
                throw KSolverExecutorTimeoutException(ex.message)
            } else {
                throw KSolverExecutorOtherException(ex)
            }
        }
    }
}
