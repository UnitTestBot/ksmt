package io.ksmt.solver.maxsmt.solvers.runner

import com.jetbrains.rd.util.lifetime.isNotAlive
import io.ksmt.KContext
import io.ksmt.runner.generated.ConfigurationBuilder
import io.ksmt.runner.generated.createConfigurationBuilder
import io.ksmt.runner.generated.models.SolverType
import io.ksmt.runner.generated.solverType
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.runner.KSolverRunnerManager
import kotlin.reflect.KClass
import kotlin.time.Duration

open class KMaxSMTSolverRunnerManager(
    workerPoolSize: Int = DEFAULT_WORKER_POOL_SIZE,
    hardTimeout: Duration = DEFAULT_HARD_TIMEOUT,
    workerProcessIdleTimeout: Duration = DEFAULT_WORKER_PROCESS_IDLE_TIMEOUT
) : KSolverRunnerManager(workerPoolSize, hardTimeout, workerProcessIdleTimeout) {
    fun <C : KSolverConfiguration> createMaxSMTSolver(
        ctx: KContext,
        maxSmtCtx: KMaxSMTContext,
        solver: KClass<out KSolver<C>>
    ): KMaxSMTSolverRunner<C> {
        if (workers.lifetime.isNotAlive) {
            throw KSolverException("Solver runner manager is terminated")
        }
        val solverType = solver.solverType
        if (solverType != SolverType.Custom) {
            return KMaxSMTSolverRunner(
                this, ctx, maxSmtCtx, solverType.createConfigurationBuilder(), solverType
            )
        }

        return createCustomMaxSMTSolver(ctx, maxSmtCtx, solver)
    }

    private fun <C : KSolverConfiguration> createCustomMaxSMTSolver(
        ctx: KContext,
        maxSmtCtx: KMaxSMTContext,
        solver: KClass<out KSolver<C>>
    ): KMaxSMTSolverRunner<C> {
        val solverInfo = customSolvers[solver.java.name]
            ?: error("Solver $solver was not registered")

        val configurationBuilderCreator = customSolversConfiguration.getOrPut(solverInfo) {
            createConfigConstructor(solverInfo.configurationQualifiedName)
        }

        @Suppress("UNCHECKED_CAST")
        return KMaxSMTSolverRunner(
            manager = this,
            ctx = ctx,
            maxSmtCtx,
            configurationBuilder = configurationBuilderCreator as ConfigurationBuilder<C>,
            solverType = SolverType.Custom,
            customSolverInfo = solverInfo
        )
    }
}

// TODO: fix as it's copy-pasted from ksmt-runner SolverUtils file.
internal fun createConfigConstructor(
    configQualifiedName: String
): (KSolverUniversalConfigurationBuilder) -> KSolverConfiguration {
    val cls = Class.forName(configQualifiedName)
    val ctor = cls.getConstructor(KSolverUniversalConfigurationBuilder::class.java)
    return { builder: KSolverUniversalConfigurationBuilder -> ctor.newInstance(builder) as KSolverConfiguration }
}
