package org.ksmt.solver.runner

import kotlinx.coroutines.runBlocking
import org.ksmt.KContext
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.core.KsmtWorkerFactory
import org.ksmt.runner.core.KsmtWorkerPool
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.generated.SolverProtocolModel
import org.ksmt.runner.generated.SolverType
import org.ksmt.solver.KSolver
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.z3.KZ3Solver
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

    var active = true
        private set

    override fun close() {
        if (!active) return
        active = false
        workers.close()
    }

    private fun ensureActive() {
        check(active) { "Solver manager is already closed" }
    }

    fun createSolver(ctx: KContext, solver: KClass<out KSolver>): KSolverRunner = runBlocking {
        createSolverAsync(ctx, solver)
    }

    suspend fun createSolverAsync(ctx: KContext, solver: KClass<out KSolver>): KSolverRunner {
        ensureActive()
        val solverType = solverTypes[solver] ?: error("Unknown solver type: $solver")
        val worker = workers.getOrCreateFreeWorker()
        worker.astSerializationCtx.initCtx(ctx)
        return KSolverRunner(hardTimeout, this, worker).also {
            it.initSolver(solverType)
        }
    }

    internal fun deleteSolver(solver: KSolverRunner) {
        solver.worker.astSerializationCtx.resetCtx()
        workers.releaseWorker(solver.worker)
    }

    internal fun terminateSolver(solver: KSolverRunner) {
        solver.worker.astSerializationCtx.resetCtx()
        workers.killWorker(solver.worker)
    }

    companion object {
        private val solverTypes = mapOf(
            KZ3Solver::class to SolverType.Z3,
            KBitwuzlaSolver::class to SolverType.Bitwuzla
        )
    }
}
