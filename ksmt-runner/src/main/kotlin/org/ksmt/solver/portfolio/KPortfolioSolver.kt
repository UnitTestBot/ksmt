package org.ksmt.solver.portfolio

import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.AtomicReference
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.launch
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
    private val solvers: List<KSolverRunner<*>>,
) : KAsyncSolver<KSolverConfiguration> {
    private val lastSuccessfulSolver = AtomicReference<KSolverRunner<*>?>(null)
    private val pendingTermination = ConcurrentLinkedQueue<KSolverRunner<*>>()

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
    ): KSolverStatus =
        solverQuery {
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

    override fun close() {
        solvers.forEach { solver ->
            solverOperationScope.launch {
                solver.deleteSolverAsync()
            }
        }
    }

    private suspend inline fun solverOperation(
        crossinline block: suspend KSolverRunner<*>.() -> Unit
    ) {
        val result = awaitFirstSolverOrNull(block) { true }
        if (result is SolverOperationFailure<*>) {
            // throw exception if all solvers in portfolio failed with exception
            result.findSuccessOrThrow()
        }
    }

    private suspend inline fun solverQuery(
        crossinline block: suspend KSolverRunner<*>.() -> KSolverStatus
    ): KSolverStatus {
        terminateIfNeeded()

        lastSuccessfulSolver.getAndSet(null)

        val result = awaitFirstSolverOrNull(block) { it != KSolverStatus.UNKNOWN }

        val (solver, status) = when (result) {
            is SolverOperationSuccess -> result.solver to result.result
            /**
             * All solvers finished with Unknown or failed with exception.
             * If some solver ends up with Unknown we can treat this result as successful.
             * */
            is SolverOperationFailure -> result.findSuccessOrThrow().let { it.solver to it.result }
        }

        lastSuccessfulSolver.getAndSet(solver)

        solvers.filter { it != solver }.forEach { failedSolver ->
            solverOperationScope.launch {
                failedSolver.interruptAsync()
            }
            pendingTermination.offer(failedSolver)
        }

        return status
    }

    private suspend inline fun <T> awaitFirstSolverOrNull(
        crossinline operation: suspend KSolverRunner<*>.() -> T,
        crossinline predicate: (T) -> Boolean
    ): SolverOperationResult<T> {
        val pendingSolvers = AtomicInteger(solvers.size)
        val results = ConcurrentLinkedQueue<Pair<KSolverRunner<*>, Result<T>>>()
        val resultFuture = CompletableDeferred<SolverOperationResult<T>>()
        solvers.forEach { solver ->
            solverOperationScope.launch {
                try {
                    val operationResult = solver.operation()
                    results.offer(solver to Result.success(operationResult))

                    if (predicate(operationResult)) {
                        val successResult = SolverOperationSuccess(solver, operationResult)
                        resultFuture.complete(successResult)
                    }
                } catch (ex: KSolverException) {
                    results.offer(solver to Result.failure(ex))
                } finally {
                    val pending = pendingSolvers.decrementAndGet()
                    if (pending == 0) {
                        val failure = SolverOperationFailure(results)
                        resultFuture.complete(failure)
                    }
                }
            }
        }
        return resultFuture.await()
    }

    private fun <T> SolverOperationFailure<T>.findSuccessOrThrow(): SolverOperationSuccess<T> {
        val exceptions = arrayListOf<Throwable>()
        results.forEach { result ->
            result.second
                .onSuccess { successResult ->
                    return SolverOperationSuccess(result.first, successResult)
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

    private sealed interface SolverOperationResult<T>

    private class SolverOperationSuccess<T>(
        val solver: KSolverRunner<*>,
        val result: T
    ) : SolverOperationResult<T>

    private class SolverOperationFailure<T>(
        val results: Collection<Pair<KSolverRunner<*>, Result<T>>>
    ) : SolverOperationResult<T>
}
