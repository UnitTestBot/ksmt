package org.ksmt.solver.runner

import com.jetbrains.rd.util.AtomicReference
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.generated.ConfigurationBuilder
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.async.KAsyncSolver
import org.ksmt.solver.runner.KSolverRunnerManager.CustomSolverInfo
import org.ksmt.sort.KBoolSort
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
        runBlocking {
            deleteSolverAsync()
        }
    }

    override suspend fun configureAsync(configurator: Config.() -> Unit) {
        val universalConfigurator = KSolverRunnerUniversalConfigurator()
        configurationBuilder(universalConfigurator).configurator()
        val config = universalConfigurator.config

        try {
            ensureInitializedAndExecute(onException = {}) {
                configureAsync(config)
            }
        } finally {
            solverState.configure(config)
        }
    }

    override suspend fun assertAsync(expr: KExpr<KBoolSort>) {
        ctx.ensureContextMatch(expr)

        try {
            ensureInitializedAndExecute(onException = {}) {
                assertAsync(expr)
            }
        } finally {
            solverState.assert(expr)
        }
    }

    override suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        ctx.ensureContextMatch(expr, trackVar)

        try {
            ensureInitializedAndExecute(onException = {}) {
                assertAndTrackAsync(expr, trackVar)
            }
        } finally {
            solverState.assertAndTrack(expr, trackVar)
        }
    }

    override suspend fun pushAsync() {
        try {
            executeIfInitialized(onException = {}) {
                pushAsync()
            }
        } finally {
            solverState.push()
        }
    }

    override suspend fun popAsync(n: UInt) {
        try {
            executeIfInitialized(onException = {}) {
                popAsync(n)
            }
        } finally {
            solverState.pop(n)
        }
    }

    override suspend fun checkAsync(timeout: Duration): KSolverStatus =
        handleCheckSatExceptionAsUnknown {
            checkAsync(timeout)
        }

    override suspend fun checkWithAssumptionsAsync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus {
        ctx.ensureContextMatch(assumptions)

        return handleCheckSatExceptionAsUnknown {
            checkWithAssumptionsAsync(assumptions, timeout)
        }
    }

    override suspend fun modelAsync(): KModel = lastSatModel.updateIfNull {
        executeIfInitialized(
            onException = { ex -> throw KSolverException("Model is not available", ex) },
            body = { modelAsync() }
        ) ?: throw KSolverException("Solver is not initialized")
    }

    override suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> = lastUnsatCore.updateIfNull {
        executeIfInitialized(
            onException = { ex -> throw KSolverException("Unsat core is not available", ex) },
            body = { unsatCoreAsync() }
        ) ?: throw KSolverException("Solver is not initialized")
    }

    override suspend fun reasonOfUnknownAsync(): String = lastReasonOfUnknown.updateIfNull {
        executeIfInitialized(
            onException = { ex -> throw KSolverException("Reason of unknown is not available", ex) },
            body = { reasonOfUnknownAsync() }
        ) ?: throw KSolverException("Solver is not initialized")
    }

    override suspend fun interruptAsync() {
        executeIfInitialized(onException = {}) {
            interruptAsync()
        }
    }

    suspend fun deleteSolverAsync() {
        isActive.set(false)
        executorInitializationLock.withLock {
            val executor = executorRef.getAndSet(null)
            executor?.let { runOnExecutor(it, onException = { }) { deleteSolver() } }
        }
    }

    internal fun terminateSolverIfBusy() {
        executorRef.get()?.terminateIfBusy()
    }

    private suspend inline fun <T> runOnExecutor(
        executor: KSolverRunnerExecutor,
        onException: (KSolverExecutorException) -> T,
        crossinline body: suspend KSolverRunnerExecutor.() -> T
    ): T = try {
        executor.body()
    } catch (ex: KSolverExecutorException) {
        executorRef.compareAndSet(executor, null)
        executor.terminate()
        onException(ex)
    }

    private suspend inline fun <T> ensureInitializedAndExecute(
        onException: (KSolverExecutorException) -> T,
        crossinline body: suspend KSolverRunnerExecutor.() -> T
    ): T {
        val executor = executorRef.get()
        if (executor != null) {
            return runOnExecutor(executor, onException, body)
        }

        val freshExecutor = try {
            executorInitializationLock.withLock {
                executorRef.updateIfNull {
                    initExecutor()
                }
            }
        } catch (ex: KSolverExecutorException) {
            executorRef.reset()
            return onException(ex)
        }

        return runOnExecutor(freshExecutor, onException, body)
    }

    private suspend inline fun <T> executeIfInitialized(
        onException: (KSolverExecutorException) -> T,
        crossinline body: suspend KSolverRunnerExecutor.() -> T
    ): T? {
        val executor = executorRef.get()
        return executor?.let { runOnExecutor(it, onException, body) }
    }

    private suspend fun initExecutor(): KSolverRunnerExecutor {
        if (!isActive.get()) {
            throw KSolverExecutorNotAliveException()
        }
        val executor = manager.createSolverExecutor(ctx, solverType, customSolverInfo)
        solverState.apply(executor)
        return executor
    }

    private suspend inline fun handleCheckSatExceptionAsUnknown(
        crossinline body: suspend KSolverRunnerExecutor.() -> KSolverStatus
    ): KSolverStatus {
        lastReasonOfUnknown.reset()
        lastSatModel.reset()
        lastUnsatCore.reset()

        return ensureInitializedAndExecute(
            body = body,
            onException = { ex ->
                if (ex is KSolverExecutorTimeoutException) {
                    lastReasonOfUnknown.getAndSet("timeout: ${ex.message}")
                } else {
                    lastReasonOfUnknown.getAndSet("error: $ex")
                }
                KSolverStatus.UNKNOWN
            }
        )
    }

    private fun <T> AtomicReference<T?>.reset() {
        getAndSet(null)
    }

    private suspend inline fun <T> AtomicReference<T?>.updateIfNull(
        crossinline body: suspend () -> T
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
}
