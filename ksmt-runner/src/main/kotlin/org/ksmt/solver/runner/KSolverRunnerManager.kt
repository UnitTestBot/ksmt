package org.ksmt.solver.runner

import com.jetbrains.rd.util.lifetime.isNotAlive
import org.ksmt.KContext
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.core.KsmtWorkerFactory
import org.ksmt.runner.core.KsmtWorkerPool
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.core.WorkerInitializationFailedException
import org.ksmt.runner.generated.createConfigurationBuilder
import org.ksmt.runner.generated.models.SolverProtocolModel
import org.ksmt.runner.generated.models.SolverType
import org.ksmt.runner.generated.solverType
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverException
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

open class KSolverRunnerManager(
    workerPoolSize: Int = 1,
    private val hardTimeout: Duration = 10.seconds,
    workerProcessIdleTimeout: Duration = 100.seconds
) : AutoCloseable {
    private val workers = KsmtWorkerPool(
        maxWorkerPoolSize = workerPoolSize,
        initializationTimeout = 15.seconds,
        workerProcessIdleTimeout = workerProcessIdleTimeout,
        workerFactory = object : KsmtWorkerFactory<SolverProtocolModel> {
            override val childProcessEntrypoint = KSolverWorkerProcess::class
            override fun mkWorker(id: Int, process: RdServer) = KSolverWorker(id, process)
            override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
        }
    )

    override fun close() {
        workers.terminate()
    }

    fun <C : KSolverConfiguration> createSolver(
        ctx: KContext,
        solver: KClass<out KSolver<C>>
    ): KSolverRunner<C> {
        if (workers.lifetime.isNotAlive) {
            throw KSolverException("Solver runner manager is terminated")
        }
        val solverType = solver.solverType
        return KSolverRunner(this, ctx, solverType.createConfigurationBuilder(), solverType)
    }

    internal suspend fun createSolverExecutor(ctx: KContext, solverType: SolverType): KSolverRunnerExecutor {
        val worker = try {
            workers.getOrCreateFreeWorker()
        } catch (ex: WorkerInitializationFailedException) {
            throw KSolverExecutorWorkerInitializationException(ex)
        }
        worker.astSerializationCtx.initCtx(ctx)
        worker.lifetime.onTermination { worker.astSerializationCtx.resetCtx() }
        return KSolverRunnerExecutor(hardTimeout, worker).also {
            it.initSolver(solverType)
        }
    }
}
