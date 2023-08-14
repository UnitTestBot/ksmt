package io.ksmt.solver.portfolio

import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.AtomicReference
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineExceptionHandler
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.Job
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.runBlocking
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.async.KAsyncSolver
import io.ksmt.solver.runner.KSolverRunner
import io.ksmt.sort.KBoolSort
import java.util.concurrent.ConcurrentLinkedQueue
import java.util.concurrent.atomic.AtomicBoolean
import java.util.concurrent.atomic.AtomicReferenceArray
import kotlin.reflect.KClass
import kotlin.time.Duration

class KPortfolioSolver(
    solverRunners: List<Pair<KClass<out KSolver<*>>, KSolverRunner<*>>>
) : KAsyncSolver<KSolverConfiguration> {
    private val lastSuccessfulSolver = AtomicReference<KSolverRunner<*>?>(null)
    private val pendingTermination = ConcurrentLinkedQueue<KSolverRunner<*>>()

    @OptIn(DelicateCoroutinesApi::class)
    private inner class SolverOperationState(
        val solverId: Int,
        val solver: KSolverRunner<*>
    ) : AutoCloseable {
        val operationCompletion = AtomicReference<CompletableDeferred<Unit>?>(null)

        /**
         * Solver operation thread.
         * We maintain the following properties:
         * 1. All operation with a single solver are sequential
         * 2. Since we have a thread per solver,
         * operations with different solvers are concurrent.
         * */
        private val operationCtx = newSingleThreadContext("portfolio-solver")
        private val exceptionIgnoringCtx = operationCtx + CoroutineExceptionHandler { _, ex ->
            // Uncaught exception during solver operation -> solver is not valid
            removeSolverFromPortfolio(solverId)
            operationCompletion.get()?.completeExceptionally(ex)
            solver.close()
        }

        val operationScope = CoroutineScope(exceptionIgnoringCtx)

        override fun close() {
            operationCtx.close()
        }
    }

    class SolverStatistic(val solver: KClass<out KSolver<*>>) {
        private val numberOfBestQueries = AtomicInteger(0)
        private val solverIsActive = AtomicBoolean(true)

        val queriesBest: Int
            get() = numberOfBestQueries.get()

        val isActive: Boolean
            get() = solverIsActive.get()

        internal fun logSolverQueryBest() {
            numberOfBestQueries.incrementAndGet()
        }

        internal fun logSolverRemovedFromPortfolio() {
            solverIsActive.set(false)
        }

        override fun toString(): String =
            "${solver.simpleName}: queriesBest=${queriesBest}, isActive=${isActive}"
    }

    private val solverStates = Array(solverRunners.size) {
        SolverOperationState(it, solverRunners[it].second)
    }

    private val activeSolvers = AtomicReferenceArray(solverStates)

    private val solverStats = Array(solverRunners.size) {
        SolverStatistic(solverRunners[it].first)
    }

    /**
     * Gather current statistic on the solvers in the portfolio.
     * */
    fun solverPortfolioStats(): List<SolverStatistic> = solverStats.toList()

    override fun configure(configurator: KSolverConfiguration.() -> Unit) = runBlocking {
        configureAsync(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) = runBlocking {
        assertAsync(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = runBlocking {
        assertAndTrackAsync(expr)
    }

    override fun push() = runBlocking {
        pushAsync()
    }

    override fun pop(n: UInt) = runBlocking {
        popAsync(n)
    }

    override fun check(timeout: Duration): KSolverStatus = runBlocking {
        checkAsync(timeout)
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = runBlocking {
        checkWithAssumptionsAsync(assumptions, timeout)
    }

    override fun model(): KModel = runBlocking {
        modelAsync()
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = runBlocking {
        unsatCoreAsync()
    }

    override fun reasonOfUnknown(): String = runBlocking {
        reasonOfUnknownAsync()
    }

    override fun interrupt() = runBlocking {
        interruptAsync()
    }

    override suspend fun configureAsync(configurator: KSolverConfiguration.() -> Unit) = solverOperation {
        configureAsync(configurator)
    }

    override suspend fun assertAsync(expr: KExpr<KBoolSort>) = solverOperation {
        assertAsync(expr)
    }

    override suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>) = solverOperation {
        assertAndTrackAsync(expr)
    }

    override suspend fun pushAsync() = solverOperation {
        pushAsync()
    }

    override suspend fun popAsync(n: UInt) = solverOperation {
        popAsync(n)
    }

    override suspend fun checkAsync(timeout: Duration): KSolverStatus = solverQuery {
        checkAsync(timeout)
    }

    override suspend fun checkWithAssumptionsAsync(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = solverQuery {
        checkWithAssumptionsAsync(assumptions, timeout)
    }

    override suspend fun modelAsync(): KModel =
        lastSuccessfulSolver.get()?.modelAsync()
            ?: throw KSolverException("No check-sat result available")

    override suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> =
        lastSuccessfulSolver.get()?.unsatCoreAsync()
            ?: throw KSolverException("No check-sat result available")

    override suspend fun reasonOfUnknownAsync(): String =
        lastSuccessfulSolver.get()?.reasonOfUnknownAsync()
            ?: throw KSolverException("No check-sat result available")

    override suspend fun interruptAsync() = solverOperation {
        interruptAsync()
    }

    override fun close() = try {
        runBlocking {
            val pendingJobs = mutableListOf<Job>()
            activeSolvers.forEach { _, solverOperationState ->
                pendingJobs += solverOperationState.operationScope.launch {
                    solverOperationState.solver.terminateSolverIfBusy()
                    solverOperationState.solver.deleteSolverAsync()
                }
            }
            pendingJobs.joinAll()
        }
    } finally {
        solverStates.forEach { it.close() }
    }

    private suspend inline fun solverOperation(
        crossinline block: suspend KSolverRunner<*>.() -> Unit
    ) {
        val result = awaitFirstSolver(block) { true }
        if (result is SolverAwaitFailure<*>) {
            // throw exception if all solvers in portfolio failed with exception
            result.findSuccessOrThrow()
        }
    }

    private suspend inline fun solverQuery(
        crossinline block: suspend KSolverRunner<*>.() -> KSolverStatus
    ): KSolverStatus {
        terminateIfNeeded()

        lastSuccessfulSolver.getAndSet(null)

        val awaitResult = awaitFirstSolver(block) {
            it != KSolverStatus.UNKNOWN
        }

        val result = when (awaitResult) {
            is SolverAwaitSuccess -> awaitResult.result.also {
                solverStats[awaitResult.result.solverId].logSolverQueryBest()
            }
            /**
             * All solvers finished with Unknown or failed with exception.
             * If some solver ends up with Unknown we can treat this result as successful.
             * */
            is SolverAwaitFailure -> awaitResult.findSuccessOrThrow()
        }

        lastSuccessfulSolver.getAndSet(result.solver)

        activeSolvers.forEach { _, solverOperationState ->
            if (solverOperationState.solver != result.solver) {
                val failedSolver = solverOperationState.solver
                solverOperationState.operationScope.launch {
                    failedSolver.interruptAsync()
                }
                pendingTermination.offer(failedSolver)
            }
        }

        return result.result
    }

    /**
     * Await for the first solver to complete the [operation]
     * with a result matching the [predicate].
     * */
    @Suppress("TooGenericExceptionCaught")
    private suspend inline fun <T> awaitFirstSolver(
        crossinline operation: suspend KSolverRunner<*>.() -> T,
        crossinline predicate: (T) -> Boolean
    ): SolverAwaitResult<T> {
        val pendingSolvers = AtomicInteger(activeSolvers.length())
        val results = ConcurrentLinkedQueue<Result<SolverOperationResult<T>>>()
        val resultFuture = CompletableDeferred<SolverAwaitResult<T>>()
        activeSolvers.forEach(
            onNullValue = {
                // Solver is not active -> skip
                pendingSolvers.decrementAndGet()
            }
        ) { solverId, solverOperationState ->
            val operationCompletion = CompletableDeferred<Unit>()
            val previousOperationCompletion = solverOperationState.operationCompletion.getAndSet(operationCompletion)
            val solver = solverOperationState.solver

            solverOperationState.operationScope.launch {
                try {
                    // Ensure solver operation order
                    previousOperationCompletion?.await()

                    val operationResult = solver.operation()
                    val solverOperationResult = SolverOperationResult(solverId, solver, operationResult)
                    results.offer(Result.success(solverOperationResult))

                    if (predicate(operationResult)) {
                        val successResult = SolverAwaitSuccess(solverOperationResult)
                        resultFuture.complete(successResult)
                    }

                    operationCompletion.complete(Unit)
                } catch (ex: Throwable) {
                    // Solver has incorrect state now. Remove it from portfolio
                    removeSolverFromPortfolio(solverId)
                    operationCompletion.completeExceptionally(ex)
                    solver.deleteSolverAsync()
                    results.offer(Result.failure(ex))
                } finally {
                    /**
                     * Return [SolverAwaitFailure]  if all solvers completed with
                     * a result that didn't match the [predicate] or completed with an exception.
                     * */
                    val pending = pendingSolvers.decrementAndGet()
                    if (pending == 0) {
                        val failure = SolverAwaitFailure(results)
                        resultFuture.complete(failure)
                    }
                }
            }
        }

        // We have no active solvers in portfolio
        if (pendingSolvers.get() == 0) {
            val failure = SolverAwaitFailure(results)
            resultFuture.complete(failure)
        }

        return resultFuture.await()
    }

    private fun removeSolverFromPortfolio(solverId: Int) {
        activeSolvers.set(solverId, null)
        solverStats[solverId].logSolverRemovedFromPortfolio()
    }

    private fun <T> SolverAwaitFailure<T>.findSuccessOrThrow(): SolverOperationResult<T> {
        val exceptions = arrayListOf<Throwable>()
        results.forEach { result ->
            result
                .onSuccess { successResult ->
                    return successResult
                }
                .onFailure { ex -> exceptions.add(ex) }
        }
        throw KSolverException("Portfolio solver failed").also { ex ->
            exceptions.forEach { ex.addSuppressed(it) }
        }
    }

    private fun terminateIfNeeded() {
        do {
            val solver = pendingTermination.poll()
            solver?.terminateSolverIfBusy()
        } while (solver != null)
    }

    data class SolverOperationResult<T>(
        val solverId: Int,
        val solver: KSolverRunner<*>,
        val result: T
    )

    private sealed interface SolverAwaitResult<T>

    private class SolverAwaitSuccess<T>(
        val result: SolverOperationResult<T>
    ) : SolverAwaitResult<T>

    private class SolverAwaitFailure<T>(
        val results: Collection<Result<SolverOperationResult<T>>>
    ) : SolverAwaitResult<T>

    private inline fun <T> AtomicReferenceArray<T?>.forEach(
        onNullValue: (Int) -> Unit = {},
        notNullBody: (Int, T) -> Unit
    ) {
        for (i in 0 until length()) {
            val value = get(i)
            if (value == null) {
                onNullValue(i)
                continue
            }
            notNullBody(i, value)
        }
    }
}
