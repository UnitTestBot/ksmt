package io.ksmt.solver.runner

import com.jetbrains.rd.util.AtomicReference
import com.jetbrains.rd.util.threading.SpinWait
import kotlinx.coroutines.sync.Mutex
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.runner.generated.ConfigurationBuilder
import io.ksmt.runner.generated.models.SolverConfigurationParam
import io.ksmt.runner.generated.models.SolverType
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.async.KAsyncSolver
import io.ksmt.solver.runner.KSolverRunnerManager.CustomSolverInfo
import io.ksmt.sort.KBoolSort
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.time.Duration

/**
 * Stateful remote solver runner.
 *
 * Manages remote solver executor and can fully restore
 * its state after failures (e.g. hard timeout) to allow incremental usage.
 * */
class KSolverRunner<Config : KSolverConfiguration>(
    private val manager: KSolverRunnerManager,
    private val ctx: KContext,
    private val configurationBuilder: ConfigurationBuilder<Config>,
    private val solverType: SolverType,
    private val customSolverInfo: CustomSolverInfo? = null,
) : KAsyncSolver<Config> {
    private val isActive = AtomicBoolean(true)
    private val executorInitializationLock = Mutex()
    private val executorRef = AtomicReference<KSolverRunnerExecutor?>(null)

    private val lastReasonOfUnknown = AtomicReference<String?>(null)
    private val lastSatModel = AtomicReference<KModel?>(null)
    private val lastUnsatCore = AtomicReference<List<KExpr<KBoolSort>>?>(null)

    private val solverState = KSolverState()

    override fun close() {
        deleteSolverSync()
    }

    override suspend fun configureAsync(configurator: Config.() -> Unit) =
        configure(configurator) { config ->
            ensureInitializedAndExecuteAsync(onException = {}) {
                configureAsync(config)
            }
        }

    override fun configure(configurator: Config.() -> Unit) =
        configure(configurator) { config ->
            ensureInitializedAndExecuteSync(onException = {}) {
                configureSync(config)
            }
        }

    private inline fun configure(
        configurator: Config.() -> Unit,
        execute: (List<SolverConfigurationParam>) -> Unit
    ) {
        val universalConfigurator = KSolverRunnerUniversalConfigurator()
        configurationBuilder(universalConfigurator).configurator()
        val config = universalConfigurator.config

        try {
            execute(config)
        } finally {
            solverState.configure(config)
        }
    }

    override suspend fun assertAsync(expr: KExpr<KBoolSort>) =
        assert(expr) { e ->
            ensureInitializedAndExecuteAsync(onException = {}) {
                assertAsync(e)
            }
        }

    override fun assert(expr: KExpr<KBoolSort>) =
        assert(expr) { e ->
            ensureInitializedAndExecuteSync(onException = {}) {
                assertSync(e)
            }
        }

    private inline fun assert(
        expr: KExpr<KBoolSort>,
        execute: (KExpr<KBoolSort>) -> Unit
    ) {
        ctx.ensureContextMatch(expr)

        try {
            execute(expr)
        } finally {
            solverState.assert(expr)
        }
    }

    override suspend fun assertAsync(exprs: List<KExpr<KBoolSort>>) =
        bulkAssert(exprs) { e ->
            ensureInitializedAndExecuteAsync(onException = {}) {
                bulkAssertAsync(e)
            }
        }

    override fun assert(exprs: List<KExpr<KBoolSort>>) =
        bulkAssert(exprs) { e ->
            ensureInitializedAndExecuteSync(onException = {}) {
                bulkAssertSync(e)
            }
        }

    private inline fun bulkAssert(
        exprs: List<KExpr<KBoolSort>>,
        execute: (List<KExpr<KBoolSort>>) -> Unit
    ) {
        ctx.ensureContextMatch(exprs)

        try {
            execute(exprs)
        } finally {
            exprs.forEach { solverState.assert(it) }
        }
    }

    override suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>) =
        assertAndTrack(expr) { e ->
            ensureInitializedAndExecuteAsync(onException = {}) {
                assertAndTrackAsync(e)
            }
        }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) =
        assertAndTrack(expr) { e ->
            ensureInitializedAndExecuteSync(onException = {}) {
                assertAndTrackSync(e)
            }
        }

    private inline fun assertAndTrack(
        expr: KExpr<KBoolSort>,
        execute: (KExpr<KBoolSort>) -> Unit
    ) {
        ctx.ensureContextMatch(expr)

        try {
            execute(expr)
        } finally {
            solverState.assertAndTrack(expr)
        }
    }

    override suspend fun assertAndTrackAsync(exprs: List<KExpr<KBoolSort>>) =
        bulkAssertAndTrack(exprs) { e ->
            ensureInitializedAndExecuteAsync(onException = {}) {
                bulkAssertAndTrackAsync(e)
            }
        }

    override fun assertAndTrack(exprs: List<KExpr<KBoolSort>>) =
        bulkAssertAndTrack(exprs) { e ->
            ensureInitializedAndExecuteSync(onException = {}) {
                bulkAssertAndTrackSync(e)
            }
        }

    private inline fun bulkAssertAndTrack(
        exprs: List<KExpr<KBoolSort>>,
        execute: (List<KExpr<KBoolSort>>) -> Unit
    ) {
        ctx.ensureContextMatch(exprs)

        try {
            execute(exprs)
        } finally {
            exprs.forEach { solverState.assertAndTrack(it) }
        }
    }

    override suspend fun pushAsync() = push {
        executeIfInitialized(onException = {}) {
            pushAsync()
        }
    }

    override fun push() = push {
        executeIfInitialized(onException = {}) {
            pushSync()
        }
    }

    private inline fun push(execute: () -> Unit) {
        try {
            execute()
        } finally {
            solverState.push()
        }
    }

    override suspend fun popAsync(n: UInt) = pop(n) {
        executeIfInitialized(onException = {}) {
            popAsync(it)
        }
    }

    override fun pop(n: UInt) = pop(n) {
        executeIfInitialized(onException = {}) {
            popSync(it)
        }
    }

    private inline fun pop(n: UInt, execute: (UInt) -> Unit) {
        try {
            execute(n)
        } finally {
            solverState.pop(n)
        }
    }

    override suspend fun checkAsync(timeout: Duration): KSolverStatus =
        handleCheckSatExceptionAsUnknownAsync {
            checkAsync(timeout)
        }

    override fun check(timeout: Duration): KSolverStatus =
        handleCheckSatExceptionAsUnknownSync {
            checkSync(timeout)
        }

    override suspend fun checkWithAssumptionsAsync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = checkWithAssumptions(assumptions) {
        handleCheckSatExceptionAsUnknownAsync {
            checkWithAssumptionsAsync(assumptions, timeout)
        }
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = checkWithAssumptions(assumptions) {
        handleCheckSatExceptionAsUnknownSync {
            checkWithAssumptionsSync(assumptions, timeout)
        }
    }

    private inline fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        execute: () -> KSolverStatus
    ): KSolverStatus {
        ctx.ensureContextMatch(assumptions)

        return execute()
    }

    override suspend fun modelAsync(): KModel = model { modelAsync() }

    override fun model(): KModel = model { modelSync() }

    private inline fun model(
        execute: KSolverRunnerExecutor.() -> KModel
    ): KModel = lastSatModel.updateIfNull {
        executeIfInitialized(
            onException = { ex -> throw KSolverException("Model is not available", ex) },
            body = { execute() }
        ) ?: throw KSolverException("Solver is not initialized")
    }

    override suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> =
        unsatCore { unsatCoreAsync() }

    override fun unsatCore(): List<KExpr<KBoolSort>> =
        unsatCore { unsatCoreSync() }

    private inline fun unsatCore(
        execute: KSolverRunnerExecutor.() -> List<KExpr<KBoolSort>>
    ): List<KExpr<KBoolSort>> = lastUnsatCore.updateIfNull {
        executeIfInitialized(
            onException = { ex -> throw KSolverException("Unsat core is not available", ex) },
            body = { execute() }
        ) ?: throw KSolverException("Solver is not initialized")
    }

    override suspend fun reasonOfUnknownAsync(): String =
        reasonOfUnknown { reasonOfUnknownAsync() }

    override fun reasonOfUnknown(): String =
        reasonOfUnknown { reasonOfUnknownSync() }

    private inline fun reasonOfUnknown(
        execute: KSolverRunnerExecutor.() -> String
    ): String = lastReasonOfUnknown.updateIfNull {
        executeIfInitialized(
            onException = { ex -> throw KSolverException("Reason of unknown is not available", ex) },
            body = { execute() }
        ) ?: throw KSolverException("Solver is not initialized")
    }

    override suspend fun interruptAsync() {
        executeIfInitialized(onException = {}) {
            interruptAsync()
        }
    }

    override fun interrupt() {
        executeIfInitialized(onException = {}) {
            interruptSync()
        }
    }

    suspend fun deleteSolverAsync() = deleteSolver(
        acquireLock = { lock() },
        deleteSolver = { deleteSolverAsync() }
    )

    fun deleteSolverSync() = deleteSolver(
        acquireLock = { lockSync() },
        deleteSolver = { deleteSolverSync() }
    )

    private inline fun deleteSolver(
        acquireLock: Mutex.() -> Unit,
        deleteSolver: KSolverRunnerExecutor.() -> Unit,
    ) {
        isActive.set(false)
        executorInitializationLock.withLock({ acquireLock() }) {
            val executor = executorRef.getAndSet(null)
            executor?.let {
                runOnExecutor(it, onException = { }) { deleteSolver() }
            }
        }
    }

    internal fun terminateSolverIfBusy() {
        executorRef.get()?.terminateIfBusy()
    }

    private suspend inline fun <T> ensureInitializedAndExecuteAsync(
        onException: (KSolverExecutorException) -> T,
        crossinline body: suspend KSolverRunnerExecutor.() -> T
    ): T = ensureInitializedAndExecute(
        onException = { onException(it) },
        body = { body() },
        acquireLock = { lock() },
        initExecutor = { initExecutorAsync() }
    )

    private inline fun <T> ensureInitializedAndExecuteSync(
        onException: (KSolverExecutorException) -> T,
        body: KSolverRunnerExecutor.() -> T
    ): T = ensureInitializedAndExecute(
        onException = { onException(it) },
        body = { body() },
        acquireLock = { lockSync() },
        initExecutor = { initExecutorSync() }
    )

    private inline fun <T> ensureInitializedAndExecute(
        onException: (KSolverExecutorException) -> T,
        body: KSolverRunnerExecutor.() -> T,
        acquireLock: Mutex.() -> Unit,
        initExecutor: () -> KSolverRunnerExecutor
    ): T {
        val executor = executorRef.get()
        if (executor != null) {
            return runOnExecutor(executor, onException) { body() }
        }

        val freshExecutor = try {
            executorInitializationLock.withLock({ acquireLock() }) {
                executorRef.updateIfNull {
                    initExecutor()
                }
            }
        } catch (ex: KSolverExecutorException) {
            executorRef.reset()
            return onException(ex)
        }

        return runOnExecutor(freshExecutor, onException) { body() }
    }

    private suspend fun initExecutorAsync(): KSolverRunnerExecutor = initExecutor {
        manager.createSolverExecutorAsync(ctx, solverType, customSolverInfo).also {
            solverState.applyAsync(it)
        }
    }

    private fun initExecutorSync(): KSolverRunnerExecutor = initExecutor {
        manager.createSolverExecutorSync(ctx, solverType, customSolverInfo).also {
            solverState.applySync(it)
        }
    }

    private inline fun initExecutor(
        createAndInitExecutor: () -> KSolverRunnerExecutor
    ): KSolverRunnerExecutor {
        if (!isActive.get()) {
            throw KSolverExecutorNotAliveException()
        }
        return createAndInitExecutor()
    }

    private suspend inline fun handleCheckSatExceptionAsUnknownAsync(
        crossinline body: suspend KSolverRunnerExecutor.() -> KSolverStatus
    ): KSolverStatus = handleCheckSatExceptionAsUnknown { onException ->
        ensureInitializedAndExecuteAsync(
            body = body,
            onException = { ex -> onException(ex) }
        )
    }

    private inline fun handleCheckSatExceptionAsUnknownSync(
        crossinline body: KSolverRunnerExecutor.() -> KSolverStatus
    ): KSolverStatus = handleCheckSatExceptionAsUnknown { onException ->
        ensureInitializedAndExecuteSync(
            body = body,
            onException = { ex -> onException(ex) }
        )
    }

    private inline fun handleCheckSatExceptionAsUnknown(
        execute: ((KSolverExecutorException) -> KSolverStatus) -> KSolverStatus
    ): KSolverStatus {
        lastReasonOfUnknown.reset()
        lastSatModel.reset()
        lastUnsatCore.reset()

        return execute { ex ->
            if (ex is KSolverExecutorTimeoutException) {
                lastReasonOfUnknown.getAndSet("timeout: ${ex.message}")
            } else {
                lastReasonOfUnknown.getAndSet("error: $ex")
            }
            KSolverStatus.UNKNOWN
        }
    }

    private inline fun <T> runOnExecutor(
        executor: KSolverRunnerExecutor,
        onException: (KSolverExecutorException) -> T,
        body: KSolverRunnerExecutor.() -> T
    ): T = try {
        executor.body()
    } catch (ex: KSolverExecutorException) {
        executorRef.compareAndSet(executor, null)
        executor.terminate()
        onException(ex)
    }

    private inline fun <T> executeIfInitialized(
        onException: (KSolverExecutorException) -> T,
        body: KSolverRunnerExecutor.() -> T
    ): T? {
        val executor = executorRef.get()
        return executor?.let { runOnExecutor(it, onException) { body() } }
    }

    private fun <T> AtomicReference<T?>.reset() {
        getAndSet(null)
    }

    private inline fun <T> AtomicReference<T?>.updateIfNull(
        body: () -> T
    ): T {
        val oldValue = get()
        if (oldValue != null) return oldValue

        val newValue = body()
        if (compareAndSet(null, newValue)) {
            return newValue
        }

        while (true) {
            val value = get()
            if (value != null) return value

            /**
             * Updated from null -> value -> null
             * According to our workflow that means:
             * <start> --> updateIfNull (current) --------------> <we are here>
             *         |                                     |
             *         -> updateIfNull (parallel) -> reset --|
             *
             * Since reset was performed we need to recompute [body].
             * */

            val updatedValue = body()
            compareAndSet(null, updatedValue)
        }
    }

    private fun Mutex.lockSync() {
        SpinWait.spinUntil { tryLock() }
    }

    private inline fun <T> Mutex.withLock(
        acquireLock: Mutex.() -> Unit,
        action: () -> T
    ): T = try {
        acquireLock()
        action()
    } finally {
        unlock()
    }
}
