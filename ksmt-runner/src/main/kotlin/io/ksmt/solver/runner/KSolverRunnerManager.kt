package io.ksmt.solver.runner

import com.jetbrains.rd.util.lifetime.isNotAlive
import io.ksmt.KContext
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.core.KsmtWorkerFactory
import io.ksmt.runner.core.KsmtWorkerPool
import io.ksmt.runner.core.KsmtWorkerSession
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.core.WorkerInitializationFailedException
import io.ksmt.runner.generated.ConfigurationBuilder
import io.ksmt.runner.generated.createConfigConstructor
import io.ksmt.runner.generated.createConfigurationBuilder
import io.ksmt.runner.generated.models.SolverProtocolModel
import io.ksmt.runner.generated.models.SolverType
import io.ksmt.runner.generated.solverType
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.maxsmt.KMaxSMTContext
import kotlinx.coroutines.runBlocking
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
        maxsmtCtx: KMaxSMTContext = KMaxSMTContext(),
        solver: KClass<out KSolver<C>>
    ): KSolverRunner<C> {
        if (workers.lifetime.isNotAlive) {
            throw KSolverException("Solver runner manager is terminated")
        }
        val solverType = solver.solverType
        if (solverType != SolverType.Custom) {
            return KSolverRunner(this, ctx, maxsmtCtx, solverType.createConfigurationBuilder(), solverType)
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
     * See [io.ksmt.solver.z3.KZ3Solver] from 'ksmt-z3' as an example.
     * 2. The [configurationBuilder] class must have a constructor with a single
     * parameter of type [io.ksmt.solver.KSolverUniversalConfigurationBuilder].
     * See [io.ksmt.solver.z3.KZ3SolverUniversalConfiguration] from 'ksmt-z3' as an example.
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
