package org.ksmt.solver.portfolio

import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.threading.SingleThreadScheduler
import org.ksmt.KContext
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.runner.KSolverRunnerManager
import org.ksmt.utils.uncheckedCast
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

class KPortfolioSolverManager(
    private val solvers: List<KClass<out KSolver<out KSolverConfiguration>>>,
    portfolioPoolSize: Int = 1,
    hardTimeout: Duration = 10.seconds,
    workerProcessIdleTimeout: Duration = 100.seconds
) : KSolverRunnerManager(
    workerPoolSize = portfolioPoolSize * solvers.size,
    hardTimeout = hardTimeout,
    workerProcessIdleTimeout = workerProcessIdleTimeout
) {
    init {
        require(solvers.isNotEmpty()) { "Empty solver portfolio" }
    }

    private val lifetime = LifetimeDefinition()
    private val solverOperationScheduler = SingleThreadScheduler(lifetime, "portfolio-solver")
    private val solverOperationScope = PortfolioSolverCoroutineScope(lifetime, solverOperationScheduler)

    override fun close() {
        lifetime.terminate()
        super.close()
    }

    fun createPortfolioSolver(ctx: KContext): KPortfolioSolver {
        val solverInstances = solvers.map {
            val solverType: KClass<out KSolver<KSolverConfiguration>> = it.uncheckedCast()
            createSolver(ctx, solverType)
        }
        return KPortfolioSolver(solverOperationScope, solverInstances)
    }
}
