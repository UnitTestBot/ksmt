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
    hardTimeout: Duration = 10.seconds
) : AutoCloseable {
    init {
        require(solvers.isNotEmpty()) { "Empty solver portfolio" }
    }

    private val lifetime = LifetimeDefinition()
    private val solverOperationScheduler = SingleThreadScheduler(lifetime, "portfolio-solver")
    private val solverOperationScope = PortfolioSolverCoroutineScope(lifetime, solverOperationScheduler)

    private val solverManager = KSolverRunnerManager(
        workerPoolSize = solvers.size,
        hardTimeout = hardTimeout
    )

    init {
        lifetime.onTermination { solverManager.close() }
    }

    override fun close() {
        lifetime.terminate()
    }

    fun createSolver(ctx: KContext): KPortfolioSolver {
        val solverInstances = solvers.map {
            val solverType: KClass<out KSolver<KSolverConfiguration>> = it.uncheckedCast()
            solverManager.createSolver(ctx, solverType)
        }
        return KPortfolioSolver(solverOperationScope, solverInstances)
    }

}
