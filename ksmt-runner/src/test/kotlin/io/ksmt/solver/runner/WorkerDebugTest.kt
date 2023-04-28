package io.ksmt.solver.runner

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.core.KsmtWorkerFactory
import io.ksmt.runner.core.KsmtWorkerPool
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.generated.models.ContextSimplificationMode
import io.ksmt.runner.generated.models.CreateSolverParams
import io.ksmt.runner.generated.models.SolverProtocolModel
import io.ksmt.runner.generated.models.SolverType
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.time.Duration.Companion.seconds

class WorkerDebugTest {

    @Ignore
    @Test
    fun testDebuggerAttach() {
        runBlocking {
            val worker = workers.getOrCreateFreeWorker()
            worker.protocolModel.initSolver.startSuspending(
                CreateSolverParams(
                    type = SolverType.Z3,
                    contextSimplificationMode = ContextSimplificationMode.NO_SIMPLIFY,
                    customSolverQualifiedName = null,
                    customSolverConfigBuilderQualifiedName = null
                )
            )
        }
    }

    companion object {
        private lateinit var workers: KsmtWorkerPool<SolverProtocolModel>

        @BeforeAll
        @JvmStatic
        fun initSolverManager() {
            workers = KsmtWorkerPool(
                maxWorkerPoolSize = 1,
                initializationTimeout = 100.seconds,
                workerProcessIdleTimeout = 10.seconds,
                processFactory = KsmtWorkerPool.Companion::processWithDebuggerFactory,
                workerFactory = object : KsmtWorkerFactory<SolverProtocolModel> {
                    override val childProcessEntrypoint = KSolverWorkerProcess::class
                    override fun mkWorker(id: Int, process: RdServer) = KSolverWorker(id, process)
                    override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
                }
            )
        }

        @AfterAll
        @JvmStatic
        fun closeSolverManager() {
            workers.terminate()
        }
    }
}
