package io.ksmt.solver.portfolio

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.runner.KSolverRunnerManager
import io.ksmt.utils.uncheckedCast
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

class KPortfolioSolverManager(
    private val solvers: List<KClass<out KSolver<out KSolverConfiguration>>>,
    portfolioPoolSize: Int = DEFAULT_PORTFOLIO_POOL_SIZE,
    hardTimeout: Duration = DEFAULT_HARD_TIMEOUT,
    workerProcessIdleTimeout: Duration = DEFAULT_WORKER_PROCESS_IDLE_TIMEOUT
) : KSolverRunnerManager(
    workerPoolSize = portfolioPoolSize * solvers.size,
    hardTimeout = hardTimeout,
    workerProcessIdleTimeout = workerProcessIdleTimeout
) {
    init {
        require(solvers.isNotEmpty()) { "Empty solver portfolio" }
    }

    fun createPortfolioSolver(ctx: KContext): KPortfolioSolver {
        val solverInstances = solvers.map {
            val solverType: KClass<out KSolver<KSolverConfiguration>> = it.uncheckedCast()
            it to createSolver(ctx, solverType)
        }
        return KPortfolioSolver(solverInstances)
    }

    companion object {
        const val DEFAULT_PORTFOLIO_POOL_SIZE = 1
        val DEFAULT_HARD_TIMEOUT = 10.seconds
        val DEFAULT_WORKER_PROCESS_IDLE_TIMEOUT = 100.seconds
    }
}
