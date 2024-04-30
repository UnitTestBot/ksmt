package io.ksmt.solver.maxsmt.solvers.runner

import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverInterface
import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.solver.portfolio.KPortfolioSolver
import io.ksmt.solver.runner.KSolverRunner
import io.ksmt.sort.KBoolSort
import kotlinx.coroutines.runBlocking
import kotlin.reflect.KClass
import kotlin.time.Duration

class KMaxSMTPortfolioSolver(
    solverRunners: List<Pair<KClass<out KSolver<*>>, KMaxSMTSolverRunner<*>>>
) : KPortfolioSolver(solverRunners), KMaxSMTSolverInterface<KSolverConfiguration> {
    override fun assertSoft(expr: KExpr<KBoolSort>, weight: UInt) = runBlocking {
        solverOperation {
            (this as KMaxSMTSolverRunner<*>).assertSoft(expr, weight)
        }
    }

    override fun checkSubOptMaxSMT(timeout: Duration, collectStatistics: Boolean)
            : KMaxSMTResult = runBlocking {
        val solverAwaitResult = solverOperationWithResult {
            (this as KMaxSMTSolverRunner<*>).checkSubOptMaxSMT(timeout, collectStatistics)
        }

        when (solverAwaitResult) {
            is SolverAwaitSuccess -> return@runBlocking solverAwaitResult.result.result
            is SolverAwaitFailure -> throw KSolverException("MaxSMT portfolio solver failed")
        }
    }

    override fun checkMaxSMT(timeout: Duration, collectStatistics: Boolean)
            : KMaxSMTResult = runBlocking {
        val solverAwaitResult = solverOperationWithResult {
            (this as KMaxSMTSolverRunner<*>).checkMaxSMT(timeout, collectStatistics)
        }

        when (solverAwaitResult) {
            is SolverAwaitSuccess -> return@runBlocking solverAwaitResult.result.result
            is SolverAwaitFailure -> throw KSolverException("MaxSMT portfolio solver failed")
        }
    }

    override fun collectMaxSMTStatistics(): KMaxSMTStatistics = runBlocking {
        val solverAwaitResult = solverOperationWithResult {
            (this as KMaxSMTSolverRunner<*>).collectMaxSMTStatistics()
        }

        when (solverAwaitResult) {
            is SolverAwaitSuccess -> return@runBlocking solverAwaitResult.result.result
            is SolverAwaitFailure -> throw KSolverException("MaxSMT portfolio solver failed")
        }
    }

    private suspend inline fun <T> solverOperationWithResult(
        crossinline block: suspend KSolverRunner<*>.() -> T
    ): SolverAwaitResult<T> {
        val result = awaitFirstSolver(block) { true }
        if (result is SolverAwaitFailure<*>) {
            // throw exception if all solvers in portfolio failed with exception
            result.findSuccessOrThrow()
        }

        return result
    }
}
