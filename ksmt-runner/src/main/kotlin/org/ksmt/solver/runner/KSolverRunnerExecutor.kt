package org.ksmt.solver.runner

import com.jetbrains.rd.framework.impl.RdCall
import com.jetbrains.rd.framework.impl.RpcTimeouts
import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.TimeoutException
import com.jetbrains.rd.util.reactive.RdFault
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.withTimeout
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.runner.core.KsmtWorkerSession
import org.ksmt.runner.generated.models.AssertAndTrackParams
import org.ksmt.runner.generated.models.AssertParams
import org.ksmt.runner.generated.models.CheckParams
import org.ksmt.runner.generated.models.CheckWithAssumptionsParams
import org.ksmt.runner.generated.models.ContextSimplificationMode
import org.ksmt.runner.generated.models.CreateSolverParams
import org.ksmt.runner.generated.models.ModelResult
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
import org.ksmt.solver.runner.KSolverRunnerManager.CustomSolverInfo
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import kotlin.time.Duration

class KSolverRunnerExecutor(
    private val hardTimeout: Duration,
    private val worker: KsmtWorkerSession<SolverProtocolModel>,
) {
    fun configureSync(config: List<SolverConfigurationParam>) {
        ensureActive()

        queryWithTimeoutAndExceptionHandlingSync {
            configure.querySync(config)
        }
    }

    suspend fun configureAsync(config: List<SolverConfigurationParam>) {
        ensureActive()

        queryWithTimeoutAndExceptionHandlingAsync {
            configure.queryAsync(config)
        }
    }

    fun assertSync(expr: KExpr<KBoolSort>) {
        ensureActive()

        val params = AssertParams(expr)
        queryWithTimeoutAndExceptionHandlingSync {
            assert.querySync(params)
        }
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) {
        ensureActive()

        val params = AssertParams(expr)
        queryWithTimeoutAndExceptionHandlingAsync {
            assert.queryAsync(params)
        }
    }

    fun assertAndTrackSync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        ensureActive()

        val params = AssertAndTrackParams(expr, trackVar)
        queryWithTimeoutAndExceptionHandlingSync {
            assertAndTrack.querySync(params)
        }
    }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        ensureActive()

        val params = AssertAndTrackParams(expr, trackVar)
        queryWithTimeoutAndExceptionHandlingAsync {
            assertAndTrack.queryAsync(params)
        }
    }

    fun pushSync() {
        ensureActive()

        queryWithTimeoutAndExceptionHandlingSync {
            push.querySync(Unit)
        }
    }

    suspend fun pushAsync() {
        ensureActive()

        queryWithTimeoutAndExceptionHandlingAsync {
            push.queryAsync(Unit)
        }
    }

    fun popSync(n: UInt) {
        ensureActive()

        val params = PopParams(n)
        queryWithTimeoutAndExceptionHandlingSync {
            pop.querySync(params)
        }
    }

    suspend fun popAsync(n: UInt) {
        ensureActive()

        val params = PopParams(n)
        queryWithTimeoutAndExceptionHandlingAsync {
            pop.queryAsync(params)
        }
    }

    fun checkSync(timeout: Duration): KSolverStatus {
        ensureActive()

        val params = CheckParams(timeout.inWholeMilliseconds)
        val result = queryWithTimeoutAndExceptionHandlingSync {
            runCheckSatQuery {
                check.querySync(params)
            }
        }
        return result.status
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus {
        ensureActive()

        val params = CheckParams(timeout.inWholeMilliseconds)
        val result = queryWithTimeoutAndExceptionHandlingAsync {
            runCheckSatQuery {
                check.queryAsync(params)
            }
        }
        return result.status
    }

    fun checkWithAssumptionsSync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus {
        ensureActive()

        val params = CheckWithAssumptionsParams(assumptions, timeout.inWholeMilliseconds)
        val result = queryWithTimeoutAndExceptionHandlingSync {
            runCheckSatQuery {
                checkWithAssumptions.querySync(params)
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
        val result = queryWithTimeoutAndExceptionHandlingAsync {
            runCheckSatQuery {
                checkWithAssumptions.queryAsync(params)
            }
        }
        return result.status
    }

    fun modelSync(): KModel {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandlingSync {
            model.querySync(Unit)
        }
        return deserializeModel(result)
    }

    suspend fun modelAsync(): KModel {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandlingAsync {
            model.queryAsync(Unit)
        }
        return deserializeModel(result)
    }

    @Suppress("UNCHECKED_CAST")
    private fun deserializeModel(result: ModelResult): KModel {
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

    fun unsatCoreSync(): List<KExpr<KBoolSort>> {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandlingSync {
            unsatCore.querySync(Unit)
        }

        @Suppress("UNCHECKED_CAST")
        return result.core as List<KExpr<KBoolSort>>
    }

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandlingAsync {
            unsatCore.queryAsync(Unit)
        }

        @Suppress("UNCHECKED_CAST")
        return result.core as List<KExpr<KBoolSort>>
    }

    fun reasonOfUnknownSync(): String {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandlingSync {
            reasonOfUnknown.querySync(Unit)
        }
        return result.reasonUnknown
    }

    suspend fun reasonOfUnknownAsync(): String {
        ensureActive()

        val result = queryWithTimeoutAndExceptionHandlingAsync {
            reasonOfUnknown.queryAsync(Unit)
        }
        return result.reasonUnknown
    }

    fun interruptSync() {
        ensureActive()

        // No queries to interrupt
        if (!hasOngoingCheckSatQueries) {
            return
        }

        queryWithTimeoutAndExceptionHandlingSync {
            interrupt.querySync(Unit)
        }
    }

    suspend fun interruptAsync() {
        ensureActive()

        // No queries to interrupt
        if (!hasOngoingCheckSatQueries) {
            return
        }

        queryWithTimeoutAndExceptionHandlingAsync {
            interrupt.queryAsync(Unit)
        }
    }

    fun initSolverSync(solverType: SolverType, customSolverInfo: CustomSolverInfo?) {
        ensureActive()

        val params = serializeSolverInitParams(solverType, customSolverInfo)
        queryWithTimeoutAndExceptionHandlingSync {
            initSolver.querySync(params)
        }
    }

    internal suspend fun initSolverAsync(solverType: SolverType, customSolverInfo: CustomSolverInfo?) {
        ensureActive()

        val params = serializeSolverInitParams(solverType, customSolverInfo)
        queryWithTimeoutAndExceptionHandlingAsync {
            initSolver.queryAsync(params)
        }
    }

    private fun serializeSolverInitParams(
        solverType: SolverType,
        customSolverInfo: CustomSolverInfo?
    ): CreateSolverParams {
        val simplificationMode = when (worker.astSerializationCtx.ctx.simplificationMode) {
            KContext.SimplificationMode.SIMPLIFY -> ContextSimplificationMode.SIMPLIFY
            KContext.SimplificationMode.NO_SIMPLIFY -> ContextSimplificationMode.NO_SIMPLIFY
        }

        return CreateSolverParams(
            type = solverType,
            contextSimplificationMode = simplificationMode,
            customSolverQualifiedName = customSolverInfo?.solverQualifiedName,
            customSolverConfigBuilderQualifiedName = customSolverInfo?.configurationQualifiedName
        )
    }

    fun deleteSolverSync() {
        ensureActive()

        queryWithTimeoutAndExceptionHandlingSync {
            deleteSolver.querySync(Unit)
        }
        worker.release()
    }

    internal suspend fun deleteSolverAsync() {
        ensureActive()

        queryWithTimeoutAndExceptionHandlingAsync {
            deleteSolver.queryAsync(Unit)
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

    private inline fun <T> runCheckSatQuery(
        body: () -> T
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

    private val rpcHardTimout = RpcTimeouts(
        // We don't need any warnings
        warnAwaitTimeMs = RPC_TIMEOUT_MAX_VALUE_MS,
        errorAwaitTimeMs = hardTimeout.inWholeMilliseconds
    )

    private fun <TReq, Tres> RdCall<TReq, Tres>.querySync(request: TReq): Tres =
        sync(request, rpcHardTimout)

    private suspend fun <TReq, Tres> RdCall<TReq, Tres>.queryAsync(request: TReq): Tres =
        startSuspending(worker.lifetime, request)

    private suspend inline fun <T> queryWithTimeoutAndExceptionHandlingAsync(
        crossinline body: suspend SolverProtocolModel.() -> T
    ): T = queryWithTimeoutAndExceptionHandling<T, TimeoutCancellationException> {
        withTimeout(hardTimeout) {
            worker.protocolModel.body()
        }
    }

    private inline fun <T> queryWithTimeoutAndExceptionHandlingSync(
        crossinline body: SolverProtocolModel.() -> T
    ): T = queryWithTimeoutAndExceptionHandling<T, TimeoutException> {
        worker.protocolModel.body()
    }

    @Suppress(
        "TooGenericExceptionCaught",
        "SwallowedException",
        "ThrowsCount"
    )
    private inline fun <T, reified Timeout> queryWithTimeoutAndExceptionHandling(
        body: () -> T
    ): T {
        try {
            return body()
        } catch (ex: RdFault) {
            throwSolverException(ex)
        } catch (ex: Throwable) {
            terminate()
            if (ex is Timeout) {
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

    companion object {
        private const val RPC_TIMEOUT_MAX_VALUE_MS = Long.MAX_VALUE / 1_000_000
    }
}
