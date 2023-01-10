package org.ksmt.solver.runner

import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.AtomicReference
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.DelicateCoroutinesApi
import kotlinx.coroutines.launch
import kotlinx.coroutines.newSingleThreadContext
import kotlinx.coroutines.runBlocking
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import java.util.concurrent.ConcurrentLinkedQueue
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

class KPortfolioSolver(
    private val ctx: KContext,
    solvers: List<KClass<out KSolver<KSolverConfiguration>>>,
    hardTimeout: Duration = 10.seconds
) : KSolver<KSolverConfiguration> {

    @OptIn(DelicateCoroutinesApi::class)
    private val coroutineContext = newSingleThreadContext("portfolio-solver")
    private val coroutineScope = CoroutineScope(coroutineContext)

    private val solverManager = KSolverRunnerManager(
        workerPoolSize = solvers.size,
        hardTimeout = hardTimeout
    )

    private val solverInstances = solvers.map { solverManager.createSolver(ctx, it) }
    private val pendingTermination = ConcurrentLinkedQueue<KSolverRunner<*>>()

    private val lastSuccessfulSolver = AtomicReference<KSolverRunner<*>?>(null)

    override fun configure(configurator: KSolverConfiguration.() -> Unit) = runBlocking {
        configureAsync(configurator)
    }

    suspend fun configureAsync(configurator: KSolverConfiguration.() -> Unit) = solverOperation {
        configureAsync(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) = runBlocking {
        assertAsync(expr)
    }

    suspend fun assertAsync(expr: KExpr<KBoolSort>) = solverOperation {
        assertAsync(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) = runBlocking {
        assertAndTrackAsync(expr, trackVar)
    }

    suspend fun assertAndTrackAsync(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) = solverOperation {
        assertAndTrackAsync(expr, trackVar)
    }

    override fun push() = runBlocking {
        pushAsync()
    }

    suspend fun pushAsync() = solverOperation {
        pushAsync()
    }

    override fun pop(n: UInt) = runBlocking {
        popAsync(n)
    }

    suspend fun popAsync(n: UInt) = solverOperation {
        popAsync(n)
    }

    override fun check(timeout: Duration): KSolverStatus = runBlocking {
        checkAsync(timeout)
    }

    suspend fun checkAsync(timeout: Duration): KSolverStatus = solverQuery {
        checkAsync(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        runBlocking {
            checkWithAssumptionsAsync(assumptions, timeout)
        }

    suspend fun checkWithAssumptionsAsync(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        solverQuery {
            checkWithAssumptionsAsync(assumptions, timeout)
        }

    override fun model(): KModel = runBlocking {
        modelAsync()
    }

    suspend fun modelAsync(): KModel =
        lastSuccessfulSolver.get()?.modelAsync()
            ?: throw KSolverException("No check-sat result available")

    override fun unsatCore(): List<KExpr<KBoolSort>> = runBlocking {
        unsatCoreAsync()
    }

    suspend fun unsatCoreAsync(): List<KExpr<KBoolSort>> =
        lastSuccessfulSolver.get()?.unsatCoreAsync()
            ?: throw KSolverException("No check-sat result available")

    override fun reasonOfUnknown(): String = runBlocking {
        reasonOfUnknownAsync()
    }

    suspend fun reasonOfUnknownAsync(): String =
        lastSuccessfulSolver.get()?.reasonOfUnknownAsync()
            ?: throw KSolverException("No check-sat result available")

    override fun interrupt() = runBlocking {
        interruptAsync()
    }

    suspend fun interruptAsync() = solverOperation {
        interruptAsync()
    }

    override fun close() {
        coroutineContext.close()
        solverManager.close()
    }

    private suspend inline fun solverOperation(
        crossinline block: suspend KSolverRunner<*>.() -> Unit
    ) {
        awaitFirstSolverOrNull(block) { true }
    }

    private suspend inline fun solverQuery(
        crossinline block: suspend KSolverRunner<*>.() -> KSolverStatus
    ): KSolverStatus {
        terminateIfNeeded()

        lastSuccessfulSolver.getAndSet(null)

        val (solver, status) = awaitFirstSolverOrNull(block) { it != KSolverStatus.UNKNOWN }

        lastSuccessfulSolver.getAndSet(solver)

        solverInstances.filter { it != solver }.forEach { failedSolver ->
            coroutineScope.launch {
                failedSolver.interruptAsync()
            }
            pendingTermination.offer(failedSolver)
        }

        return status ?: KSolverStatus.UNKNOWN
    }

    private suspend inline fun <T> awaitFirstSolverOrNull(
        crossinline operation: suspend KSolverRunner<*>.() -> T,
        crossinline predicate: (T) -> Boolean
    ): Pair<KSolverRunner<*>, T?> {
        val pendingSolvers = AtomicInteger(solverInstances.size)
        val resultFuture = CompletableDeferred<Pair<KSolverRunner<*>, T?>>()
        solverInstances.forEach { solver ->
            coroutineScope.launch {
                val operationResult = solver.operation()
                if (predicate(operationResult)) {
                    resultFuture.complete(solver to operationResult)
                }
                val pending = pendingSolvers.decrementAndGet()
                if (pending == 0) {
                    resultFuture.complete(solver to null)
                }
            }
        }
        return resultFuture.await()
    }

    private fun terminateIfNeeded() {
        do {
            val solver = pendingTermination.poll()
            solver?.terminateSolverIfBusy()
        } while (solver != null)
    }

}
