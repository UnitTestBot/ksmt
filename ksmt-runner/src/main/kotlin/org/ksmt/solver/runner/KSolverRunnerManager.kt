package org.ksmt.solver.runner

import com.jetbrains.rd.util.lifetime.isNotAlive
import kotlinx.coroutines.runBlocking
import org.ksmt.KContext
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.core.KsmtWorkerFactory
import org.ksmt.runner.core.KsmtWorkerPool
import org.ksmt.runner.core.KsmtWorkerSession
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.core.WorkerInitializationFailedException
import org.ksmt.runner.generated.ConfigurationBuilder
import org.ksmt.runner.generated.createConfigConstructor
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
    workerPoolSize: Int = DEFAULT_WORKER_POOL_SIZE,
    private val hardTimeout: Duration = DEFAULT_HARD_TIMEOUT,
    workerProcessIdleTimeout: Duration = DEFAULT_WORKER_PROCESS_IDLE_TIMEOUT
) : AutoCloseable {
    private val workers = KsmtWorkerPool(
        maxWorkerPoolSize = workerPoolSize,
        initializationTimeout = SOLVER_WORKER_INITIALIZATION_TIMEOUT,
        workerProcessIdleTimeout = workerProcessIdleTimeout,
        workerFactory = object : KsmtWorkerFactory<SolverProtocolModel> {
            override val childProcessEntrypoint = KSolverWorkerProcess::class
            override fun mkWorker(id: Int, process: RdServer) = KSolverWorker(id, process)
            override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
        }
    )

    private val customSolvers = hashMapOf<String, CustomSolverInfo>()
    private val customSolversConfiguration = hashMapOf<CustomSolverInfo, ConfigurationBuilder<KSolverConfiguration>>()

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
        if (solverType != SolverType.Custom) {
            return KSolverRunner(this, ctx, solverType.createConfigurationBuilder(), solverType)
        }

        return createCustomSolver(ctx, solver)
    }

    private fun <C : KSolverConfiguration> createCustomSolver(
        ctx: KContext,
        solver: KClass<out KSolver<C>>
    ): KSolverRunner<C> {
        val solverInfo = customSolvers[solver.java.name]
            ?: error("Solver $solver was not registered")

        val configurationBuilderCreator = customSolversConfiguration.getOrPut(solverInfo) {
            createConfigConstructor(solverInfo.configurationQualifiedName)
        }

        @Suppress("UNCHECKED_CAST")
        return KSolverRunner(
            manager = this,
            ctx = ctx,
            configurationBuilder = configurationBuilderCreator as ConfigurationBuilder<C>,
            solverType = SolverType.Custom,
            customSolverInfo = solverInfo
        )
    }

    /**
     * Register the user-defined solver in runner and allow
     * the custom solver to run in a separate process / portfolio.
     *
     * Requirements:
     * 1. The [solver] class must have a constructor with a single parameter of type [KContext].
     * See [org.ksmt.solver.z3.KZ3Solver] from 'ksmt-z3' as an example.
     * 2. The [configurationBuilder] class must have a constructor with a single
     * parameter of type [org.ksmt.solver.KSolverUniversalConfigurationBuilder].
     * See [org.ksmt.solver.z3.KZ3SolverUniversalConfiguration] from 'ksmt-z3' as an example.
     * */
    fun <C : KSolverConfiguration> registerSolver(
        solver: KClass<out KSolver<C>>,
        configurationBuilder: KClass<out C>
    ) {
        val solverQualifiedName = solver.java.name
        val configBuilderQualifiedName = configurationBuilder.java.name

        customSolvers[solverQualifiedName] = CustomSolverInfo(solverQualifiedName, configBuilderQualifiedName)
    }

    internal suspend fun createSolverExecutorAsync(
        ctx: KContext,
        solverType: SolverType,
        customSolverInfo: CustomSolverInfo?
    ) = createSolverExecutor(
        ctx = ctx,
        solverType = solverType,
        customSolverInfo = customSolverInfo,
        getWorker = {
            workers.getOrCreateFreeWorker()
        },
        initSolverRunner = { type, info ->
            initSolverAsync(type, info)
        }
    )

    internal fun createSolverExecutorSync(
        ctx: KContext,
        solverType: SolverType,
        customSolverInfo: CustomSolverInfo?
    ) = createSolverExecutor(
        ctx = ctx,
        solverType = solverType,
        customSolverInfo = customSolverInfo,
        getWorker = {
            runBlocking { workers.getOrCreateFreeWorker() }
        },
        initSolverRunner = { type, info ->
            initSolverSync(type, info)
        }
    )

    private inline fun createSolverExecutor(
        ctx: KContext,
        solverType: SolverType,
        customSolverInfo: CustomSolverInfo?,
        getWorker: () -> KsmtWorkerSession<SolverProtocolModel>,
        initSolverRunner: KSolverRunnerExecutor.(SolverType, CustomSolverInfo?) -> Unit
    ): KSolverRunnerExecutor {
        val worker = try {
            getWorker()
        } catch (ex: WorkerInitializationFailedException) {
            throw KSolverExecutorWorkerInitializationException(ex)
        }
        worker.astSerializationCtx.initCtx(ctx)
        worker.lifetime.onTermination { worker.astSerializationCtx.resetCtx() }
        return KSolverRunnerExecutor(hardTimeout, worker).also {
            it.initSolverRunner(solverType, customSolverInfo)
        }
    }

    data class CustomSolverInfo(
        val solverQualifiedName: String,
        val configurationQualifiedName: String
    )

    companion object {
        const val DEFAULT_WORKER_POOL_SIZE = 1
        val DEFAULT_HARD_TIMEOUT = 10.seconds
        val DEFAULT_WORKER_PROCESS_IDLE_TIMEOUT = 100.seconds
        val SOLVER_WORKER_INITIALIZATION_TIMEOUT = 15.seconds
    }
}
