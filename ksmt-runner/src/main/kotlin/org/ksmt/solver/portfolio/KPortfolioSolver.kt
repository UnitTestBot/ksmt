package org.ksmt.solver.portfolio

import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.AtomicReference
import com.jetbrains.rd.util.ConcurrentHashMap
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.launch
import kotlinx.coroutines.runBlocking
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.async.KAsyncSolver
import org.ksmt.solver.runner.KSolverRunner
import org.ksmt.sort.KBoolSort
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.time.Duration

class KPortfolioSolver(
    private val solverOperationScope: CoroutineScope,
    solverRunners: List<KSolverRunner<*>>,
) : KAsyncSolver<KSolverConfiguration> {
    private val lastSuccessfulSolver = AtomicReference<KSolverRunner<*>?>(null)
    private val pendingTermination = ConcurrentLinkedQueue<KSolverRunner<*>>()
    private val solvers = ConcurrentHashMap<KSolverRunner<*>, AtomicReference<CompletableDeferred<Unit>?>>(
        solverRunners.associateWith { AtomicReference(null) }
    )

    override suspend fun configureAsync(configurator: KSolverConfiguration.() -> Unit) = solverOperation {
        configureAsync(configurator)
    }

    override suspend fun assertAsync(expr: KExpr<KBoolSort>) = solverOperation {
        assertAsync(expr)
    }

    override suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) =
        solverOperation {
            assertAndTrackAsync(expr, trackVar)
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

    override fun close() = runBlocking {
        solvers.keys.map { solver ->
            solverOperationScope.launch {
                solver.deleteSolverAsync()
            }
        }.joinAll()
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
            is SolverAwaitSuccess -> awaitResult.result
            /**
             * All solvers finished with Unknown or failed with exception.
             * If some solver ends up with Unknown we can treat this result as successful.
             * */
            is SolverAwaitFailure -> awaitResult.findSuccessOrThrow()
        }

        lastSuccessfulSolver.getAndSet(result.solver)

        solvers.keys.filter { it != result.solver }.forEach { failedSolver ->
            solverOperationScope.launch {
                failedSolver.interruptAsync()
            }
            pendingTermination.offer(failedSolver)
        }

        return result.result
    }

    @Suppress("TooGenericExceptionCaught")
    private suspend inline fun <T> awaitFirstSolver(
        crossinline operation: suspend KSolverRunner<*>.() -> T,
        crossinline predicate: (T) -> Boolean
    ): SolverAwaitResult<T> {
        val pendingSolvers = AtomicInteger(solvers.size)
        val results = ConcurrentLinkedQueue<Result<SolverOperationResult<T>>>()
        val resultFuture = CompletableDeferred<SolverAwaitResult<T>>()
        solvers.keys.forEach { solver ->
            val operationCompletion = CompletableDeferred<Unit>()
            val previousOperationCompletion = solvers[solver]?.getAndSet(operationCompletion)
            solverOperationScope.launch {
                try {
                    previousOperationCompletion?.await()
                    val operationResult = solver.operation()
                    val solverOperationResult = SolverOperationResult(solver, operationResult)
                    results.offer(Result.success(solverOperationResult))

                    if (predicate(operationResult)) {
                        val successResult = SolverAwaitSuccess(solverOperationResult)
                        resultFuture.complete(successResult)
                    }
                    operationCompletion.complete(Unit)
                } catch (ex: Throwable) {
                    // Solver has incorrect state now. Remove it from portfolio
                    solvers.remove(solver)
                    operationCompletion.completeExceptionally(ex)
                    solver.deleteSolverAsync()
                    results.offer(Result.failure(ex))
                } finally {
                    val pending = pendingSolvers.decrementAndGet()
                    if (pending == 0) {
                        val failure = SolverAwaitFailure(results)
                        resultFuture.complete(failure)
                    }
                }
            }
        }
        return resultFuture.await()
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
}
