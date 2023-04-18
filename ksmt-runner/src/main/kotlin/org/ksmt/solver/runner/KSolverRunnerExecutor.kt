package org.ksmt.solver.runner

import com.jetbrains.rd.framework.IRdTask
import com.jetbrains.rd.framework.RdTaskResult
import com.jetbrains.rd.framework.impl.RdCall
import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.TimeoutException
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.reactive.RdFault
import com.jetbrains.rd.util.threading.SynchronousScheduler
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
import org.ksmt.runner.generated.models.CheckResult
import org.ksmt.runner.generated.models.CheckWithAssumptionsParams
import org.ksmt.runner.generated.models.ContextSimplificationMode
import org.ksmt.runner.generated.models.CreateSolverParams
import org.ksmt.runner.generated.models.ModelResult
import org.ksmt.runner.generated.models.ModelEntry
import org.ksmt.runner.generated.models.ModelFuncInterpEntry
import org.ksmt.runner.generated.models.PopParams
import org.ksmt.runner.generated.models.ReasonUnknownResult
import org.ksmt.runner.generated.models.SolverConfigurationParam
import org.ksmt.runner.generated.models.SolverProtocolModel
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.runner.generated.models.UnsatCoreResult
import org.ksmt.solver.model.KFuncInterp
import org.ksmt.solver.model.KFuncInterpEntry
import org.ksmt.solver.model.KFuncInterpEntryVarsFree
import org.ksmt.solver.model.KFuncInterpEntryWithVars
import org.ksmt.solver.model.KFuncInterpVarsFree
import org.ksmt.solver.model.KFuncInterpWithVars
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
import java.util.concurrent.CompletableFuture
import java.util.concurrent.TimeUnit
import org.ksmt.utils.uncheckedCast
import kotlin.time.Duration

class KSolverRunnerExecutor(
    private val hardTimeout: Duration,
    private val worker: KsmtWorkerSession<SolverProtocolModel>,
) {
    fun configureSync(config: List<SolverConfigurationParam>) = configure(config) { cfg ->
        queryWithTimeoutAndExceptionHandlingSync {
            configure.querySync(cfg)
        }
    }

    suspend fun configureAsync(config: List<SolverConfigurationParam>) = configure(config) { cfg ->
        queryWithTimeoutAndExceptionHandlingAsync {
            configure.queryAsync(cfg)
        }
    }

    private inline fun configure(
        config: List<SolverConfigurationParam>,
        query: (List<SolverConfigurationParam>) -> Unit
    ) {
        ensureActive()
        query(config)
    }

    fun assertSync(expr: KExpr<KBoolSort>) = assert(expr) { params ->
        queryWithTimeoutAndExceptionHandlingSync {
            assert.querySync(params)
        }
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) = assert(expr) { params ->
        queryWithTimeoutAndExceptionHandlingAsync {
            assert.queryAsync(params)
        }
    }

    private inline fun assert(expr: KExpr<KBoolSort>, query: (AssertParams) -> Unit) {
        ensureActive()
        query(AssertParams(expr))
    }

    fun assertAndTrackSync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) =
        assertAndTrack(expr, trackVar) { params ->
            queryWithTimeoutAndExceptionHandlingSync {
                assertAndTrack.querySync(params)
            }
        }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) =
        assertAndTrack(expr, trackVar) { params ->
            queryWithTimeoutAndExceptionHandlingAsync {
                assertAndTrack.queryAsync(params)
            }
        }

    private inline fun assertAndTrack(
        expr: KExpr<KBoolSort>,
        trackVar: KConstDecl<KBoolSort>,
        query: (AssertAndTrackParams) -> Unit
    ) {
        ensureActive()
        query(AssertAndTrackParams(expr, trackVar))
    }

    fun pushSync() = push {
        queryWithTimeoutAndExceptionHandlingSync {
            push.querySync(Unit)
        }
    }

    suspend fun pushAsync() = push {
        queryWithTimeoutAndExceptionHandlingAsync {
            push.queryAsync(Unit)
        }
    }

    private inline fun push(query: () -> Unit) {
        ensureActive()
        query()
    }

    fun popSync(n: UInt) = pop(n) { params ->
        queryWithTimeoutAndExceptionHandlingSync {
            pop.querySync(params)
        }
    }

    suspend fun popAsync(n: UInt) = pop(n) { params ->
        queryWithTimeoutAndExceptionHandlingAsync {
            pop.queryAsync(params)
        }
    }

    private inline fun pop(n: UInt, query: (PopParams) -> Unit) {
        ensureActive()
        query(PopParams(n))
    }

    fun checkSync(timeout: Duration): KSolverStatus = check(timeout) { params ->
        queryWithTimeoutAndExceptionHandlingSync {
            runCheckSatQuery {
                check.querySync(params)
            }
        }
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus = check(timeout) { params ->
        queryWithTimeoutAndExceptionHandlingAsync {
            runCheckSatQuery {
                check.queryAsync(params)
            }
        }
    }

    private inline fun check(timeout: Duration, query: (CheckParams) -> CheckResult): KSolverStatus {
        ensureActive()

        val params = CheckParams(timeout.inWholeMilliseconds)
        val result = query(params)
        return result.status
    }

    fun checkWithAssumptionsSync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = checkWithAssumptions(assumptions, timeout) { params ->
        queryWithTimeoutAndExceptionHandlingSync {
            runCheckSatQuery {
                checkWithAssumptions.querySync(params)
            }
        }
    }

    suspend fun checkWithAssumptionsAsync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = checkWithAssumptions(assumptions, timeout) { params ->
        queryWithTimeoutAndExceptionHandlingAsync {
            runCheckSatQuery {
                checkWithAssumptions.queryAsync(params)
            }
        }
    }

    private inline fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration,
        query: (CheckWithAssumptionsParams) -> CheckResult
    ): KSolverStatus {
        ensureActive()

        val params = CheckWithAssumptionsParams(assumptions, timeout.inWholeMilliseconds)
        val result = query(params)
        return result.status
    }

    fun modelSync(): KModel = model {
        queryWithTimeoutAndExceptionHandlingSync {
            model.querySync(Unit)
        }
    }

    suspend fun modelAsync(): KModel = model {
        queryWithTimeoutAndExceptionHandlingAsync {
            model.queryAsync(Unit)
        }
    }

    private inline fun model(query: () -> ModelResult): KModel {
        ensureActive()

        val result = query()
        return deserializeModel(result)
    }

    private fun deserializeModel(result: ModelResult): KModel {
        val interpretations = result.declarations.zip(result.interpretations) { decl, interp ->
            val functionInterp = deserializeFunctionInterpretation(interp)
            (decl as KDecl<*>) to functionInterp
        }
        val uninterpretedSortUniverse = result.uninterpretedSortUniverse.associateBy(
            { entry -> entry.sort as KUninterpretedSort },
            { entry -> entry.universe.mapTo(hashSetOf()) { it as KUninterpretedSortValue } }
        )
        return KModelImpl(worker.astSerializationCtx.ctx, interpretations.toMap(), uninterpretedSortUniverse)
    }

    private fun deserializeFunctionInterpretation(interp: ModelEntry): KFuncInterp<*> {
        val decl: KDecl<KSort> = interp.decl.uncheckedCast()
        val vars: List<KDecl<*>>? = interp.vars?.uncheckedCast()
        val default: KExpr<KSort>? = interp.default?.uncheckedCast()

        val entries = interp.entries.map { deserializeFunctionInterpretationEntry(it) }
        return if (vars != null) {
            KFuncInterpWithVars(decl, vars, entries, default)
        } else {
            KFuncInterpVarsFree(decl, entries.uncheckedCast(), default)
        }
    }

    private fun deserializeFunctionInterpretationEntry(entry: ModelFuncInterpEntry): KFuncInterpEntry<KSort> {
        val args: List<KExpr<*>> = entry.args.uncheckedCast()
        val value: KExpr<KSort> = entry.value.uncheckedCast()
        return if (entry.hasVars) {
            KFuncInterpEntryWithVars.create(args, value)
        } else {
            KFuncInterpEntryVarsFree.create(args, value)
        }
    }

    fun unsatCoreSync(): List<KExpr<KBoolSort>> = unsatCore {
        queryWithTimeoutAndExceptionHandlingSync {
            unsatCore.querySync(Unit)
        }
    }

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> = unsatCore {
        queryWithTimeoutAndExceptionHandlingAsync {
            unsatCore.queryAsync(Unit)
        }
    }

    private inline fun unsatCore(query: () -> UnsatCoreResult): List<KExpr<KBoolSort>> {
        ensureActive()

        val result = query()

        @Suppress("UNCHECKED_CAST")
        return result.core as List<KExpr<KBoolSort>>
    }

    fun reasonOfUnknownSync(): String = reasonOfUnknown {
        queryWithTimeoutAndExceptionHandlingSync {
            reasonOfUnknown.querySync(Unit)
        }
    }

    suspend fun reasonOfUnknownAsync(): String = reasonOfUnknown {
        queryWithTimeoutAndExceptionHandlingAsync {
            reasonOfUnknown.queryAsync(Unit)
        }
    }

    private inline fun reasonOfUnknown(query: () -> ReasonUnknownResult): String {
        ensureActive()

        val result = query()
        return result.reasonUnknown
    }

    fun interruptSync() = interrupt {
        queryWithTimeoutAndExceptionHandlingSync {
            interrupt.querySync(Unit)
        }
    }

    suspend fun interruptAsync() = interrupt {
        queryWithTimeoutAndExceptionHandlingAsync {
            interrupt.queryAsync(Unit)
        }
    }

    private inline fun interrupt(query: () -> Unit) {
        ensureActive()

        // No queries to interrupt
        if (!hasOngoingCheckSatQueries) {
            return
        }

        query()
    }

    fun initSolverSync(solverType: SolverType, customSolverInfo: CustomSolverInfo?) =
        initSolver(solverType, customSolverInfo) { params ->
            queryWithTimeoutAndExceptionHandlingSync {
                initSolver.querySync(params)
            }
        }

    internal suspend fun initSolverAsync(solverType: SolverType, customSolverInfo: CustomSolverInfo?) =
        initSolver(solverType, customSolverInfo) { params ->
            queryWithTimeoutAndExceptionHandlingAsync {
                initSolver.queryAsync(params)
            }
        }

    private inline fun initSolver(
        solverType: SolverType,
        customSolverInfo: CustomSolverInfo?,
        query: (CreateSolverParams) -> Unit
    ) {
        ensureActive()

        val params = serializeSolverInitParams(solverType, customSolverInfo)
        query(params)
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

    fun deleteSolverSync() = deleteSolver {
        queryWithTimeoutAndExceptionHandlingSync {
            deleteSolver.querySync(Unit)
        }
    }

    internal suspend fun deleteSolverAsync() = deleteSolver {
        queryWithTimeoutAndExceptionHandlingAsync {
            deleteSolver.queryAsync(Unit)
        }
    }

    private inline fun deleteSolver(query: () -> Unit) {
        ensureActive()

        query()
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

    private fun <TReq, Tres> RdCall<TReq, Tres>.querySync(request: TReq): Tres =
        fastSync(worker.lifetime, request)

    private suspend fun <TReq, Tres> RdCall<TReq, Tres>.queryAsync(request: TReq): Tres =
        startSuspending(worker.lifetime, request)

    private fun <TReq, Tres> RdCall<TReq, Tres>.fastSync(
        lifetime: Lifetime, request: TReq
    ): Tres {
        val task = start(lifetime, request, SynchronousScheduler)
        return task.wait(hardTimeout.inWholeMilliseconds).unwrap()
    }

    /**
     * We use future instead of internal rd SpinWait based implementation
     * because usually requests don't fit into the `fast` time window, but
     * the response time is still much faster than `wait` time.
     *
     * For example, we usually see the following pattern:
     * 1. SpinWait performs 100 spins in less than 10us and forces the thread to sleep for the next 1ms.
     * 2. We receive a response in 80us.
     * 3. We are waiting for the processing of the response for the next 920us.
     * */
    private fun <T> IRdTask<T>.wait(timeoutMs: Long): RdTaskResult<T> {
        val future = CompletableFuture<RdTaskResult<T>>()
        result.advise(worker.lifetime) {
            future.complete(it)
        }
        return future.get(timeoutMs, TimeUnit.MILLISECONDS)
    }

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
}
