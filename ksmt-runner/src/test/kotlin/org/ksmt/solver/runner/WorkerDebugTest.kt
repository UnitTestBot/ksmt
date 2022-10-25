package org.ksmt.solver.runner

import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.core.KsmtWorkerFactory
import org.ksmt.runner.core.KsmtWorkerPool
import org.ksmt.runner.core.RdServer
import org.ksmt.runner.models.generated.CreateSolverParams
import org.ksmt.runner.models.generated.SolverProtocolModel
import org.ksmt.runner.models.generated.SolverType
import kotlin.test.Ignore
import kotlin.test.Test
import kotlin.time.Duration.Companion.seconds

class WorkerDebugTest {

    @Ignore
    @Test
    fun testDebuggerAttach() {
        runBlocking {
            val worker = workers.getOrCreateFreeWorker()
            worker.protocolModel.initSolver.startSuspending(CreateSolverParams(SolverType.Z3))
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
