package org.ksmt.solver.runner

import com.jetbrains.rd.util.lifetime.isNotAlive
import kotlinx.coroutines.runBlocking
import org.ksmt.KContext
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.core.KsmtWorkerFactory
import org.ksmt.runner.core.KsmtWorkerPool
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.models.generated.SolverProtocolModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

class KSolverRunnerManager(
    workerPoolSize: Int = 1,
    private val hardTimeout: Duration = 10.seconds,
    private val workerProcessIdleTimeout: Duration = 100.seconds
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

    fun createSolver(ctx: KContext, solver: KClass<out KSolver>): KSolverRunner = runBlocking {
        createSolverAsync(ctx, solver)
    }

    suspend fun createSolverAsync(ctx: KContext, solver: KClass<out KSolver>): KSolverRunner {
        if (workers.lifetime.isNotAlive) {
            throw KSolverException("Solver runner manager is terminated")
        }
        val solverType = solver.solverType
        val worker = workers.getOrCreateFreeWorker()
        worker.astSerializationCtx.initCtx(ctx)
        worker.lifetime.onTermination { worker.astSerializationCtx.resetCtx() }
        return KSolverRunner(hardTimeout, worker).also {
            it.initSolver(solverType)
        }
    }
}
