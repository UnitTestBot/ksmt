package org.ksmt.runner.core

import com.jetbrains.rd.framework.IdKind
import com.jetbrains.rd.framework.Identities
import com.jetbrains.rd.framework.Protocol
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.SocketWire
import com.jetbrains.rd.framework.util.NetUtils
import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.LogLevel
import com.jetbrains.rd.util.concurrentMapOf
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.lifetime.isAlive
import com.jetbrains.rd.util.lifetime.throwIfNotAlive
import com.jetbrains.rd.util.threading.SingleThreadScheduler
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.delay
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.ksmt.runner.core.process.NormalProcess
import org.ksmt.runner.core.process.ProcessWrapper
import org.ksmt.runner.serializer.AstSerializationCtx
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.seconds

/**
 * Pool of reusable process based workers.
 *
 * Worker phases:
 * 1. Initialization (see [createWorker])
 *
 *    Create and initialize new worker process.
 *    Add the successfully initialized worker to [readyWorkers] and
 *    notify all waiting clients (see [getOrCreateFreeWorker]).
 *
 * 2. Acquiring (see [getOrCreateFreeWorker])
 *
 *    The worker is acquired by a client and removed from [readyWorkers].
 *    If there are no workers available and the pool size is not exhausted
 *    create a new worker (see Initialization).
 *
 * 3. Release (see [releaseWorker])
 *
 *    Worker is released by client. If worker is alive return it to [readyWorkers] and
 *    notify all waiting clients. Otherwise, kill worker (see Termination).
 *
 * 4. Termination (see [killWorker])
 *
 *   The worker is released by the client, but it is not alive anymore (e.g. native exception occurred).
 *   Kill underlying process and notify all waiting clients.
 *
 * */
class KsmtWorkerPool<Model>(
    private val maxWorkerPoolSize: Int = 1,
    private val initializationTimeout: Duration = 5.seconds,
    private val checkProcessAliveDelay: Duration = 1.seconds,
    private val workerProcessIdleTimeout: Duration = 10.seconds,
    private val processFactory: (KClass<*>, List<String>) -> ProcessWrapper = ::normalProcessFactory,
    private val workerFactory: KsmtWorkerFactory<Model>
) : Lifetimed {

    /**
     * Available workers.
     *
     * [null] is used to notify waiting clients about changes in a pool size to allow
     * a new worker process to be created when possible.
     *
     * Note: In some situations, the actual size of [readyWorkers] may be
     * slightly greater than [maxWorkerPoolSize] due to nulls.
     * */
    private val readyWorkers = Channel<KsmtWorkerBase<Model>?>(capacity = Channel.UNLIMITED)

    private val workers = concurrentMapOf<Int, KsmtWorkerBase<Model>>()
    private val activeWorkers = AtomicInteger(0)
    private val workerId = AtomicInteger(1)
    override val lifetime = LifetimeDefinition()
    private val scheduler = SingleThreadScheduler(lifetime, "ksmt-runner-scheduler")
    private val coroutineScope = KsmtRdCoroutineScope(lifetime, scheduler)

    init {
        lifetime.onTermination { terminatePool() }
    }

    suspend fun getOrCreateFreeWorker(): KsmtWorkerSession<Model> {
        while (lifetime.isAlive) {
            if (activeWorkers.get() < maxWorkerPoolSize) {
                createWorker()
                continue
            }
            val worker = withTimeoutOrNull(initializationTimeout) {
                readyWorkers.receive()
            }
            if (worker != null && checkWorker(worker)) {
                val sessionLifetime = worker.lifetime.createNested()
                return KsmtWorkerSession(sessionLifetime, worker, this)
            }
        }
        throw WorkerInitializationFailedException("Worker pool is not alive")
    }

    fun releaseWorker(worker: KsmtWorkerBase<*>) {
        if (!checkWorker(worker)) return

        @Suppress("UNCHECKED_CAST")
        readyWorkers.trySendBlocking(worker as KsmtWorkerBase<Model>)
    }

    fun killWorker(worker: KsmtWorkerBase<*>) {
        worker.terminate()
    }

    override fun terminate() {
        lifetime.terminate()
    }

    private fun handleWorkerTermination(id: Int) {
        if (workers.remove(id) == null) return
        activeWorkers.decrementAndGet()

        // awake all waiting clients in getOrCreateFreeWorker
        readyWorkers.trySendBlocking(null)
    }

    private fun terminatePool() {
        readyWorkers.close()
        // keep immutable copy of workers since [killWorker] modifies [workers]
        val allWorkers = workers.values.toList()
        allWorkers.forEach { killWorker(it) }
        workers.clear()
    }

    private fun checkWorker(worker: KsmtWorkerBase<*>): Boolean {
        if (workers.containsKey(worker.id) && worker.isAlive) return true
        killWorker(worker)
        return false
    }

    private fun increaseActiveWorkersCount(): Boolean {
        val currentActiveWorkers = activeWorkers.get()
        val newValue = currentActiveWorkers + 1
        if (newValue > maxWorkerPoolSize) return false
        return activeWorkers.compareAndSet(currentActiveWorkers, newValue)
    }

    @Suppress("TooGenericExceptionCaught")
    private suspend fun createWorker() {
        if (!increaseActiveWorkersCount()) return

        val id = workerId.getAndIncrement()
        val workerLifetime = lifetime.createNested()
        val worker = try {
            initWorker(workerLifetime, id)
        } catch (ex: Exception) {
            workerLifetime.terminate()
            activeWorkers.decrementAndGet()
            throw WorkerInitializationFailedException(ex)
        }

        workers[id] = worker
        readyWorkers.send(worker)
    }

    private suspend fun initWorker(lifetime: Lifetime, id: Int): KsmtWorkerBase<Model> =
        withTimeout(initializationTimeout) {
            val rdProcess = startProcessWithRdServer(lifetime, id) { port ->
                val basicArgs = KsmtWorkerArgs(id, port, workerProcessIdleTimeout, logger.level)
                val args = workerFactory.updateArgs(basicArgs)

                processFactory(workerFactory.childProcessEntrypoint, args.toList())
            }
            val worker = workerFactory.mkWorker(id, rdProcess)
            worker.init()
            worker
        }

    private suspend fun startProcessWithRdServer(
        lifetime: Lifetime,
        id: Int,
        factory: (Int) -> ProcessWrapper
    ): RdServerProcess {
        lifetime.throwIfNotAlive()

        val port = NetUtils.findFreePort(0)

        val process = factory(port)

        return initRdServer(lifetime.createNested(), process, port, id)
    }

    private suspend fun initRdServer(
        lifetime: LifetimeDefinition,
        workerProcess: ProcessWrapper,
        port: Int,
        id: Int
    ): RdServerProcess {
        lifetime.onTermination {
            handleWorkerTermination(id)
            workerProcess.destroyForcibly()
        }

        val serializers = Serializers()
        val astSerializationCtx = AstSerializationCtx.register(serializers)
        val protocol = Protocol(
            "ksmt-server",
            serializers,
            Identities(IdKind.Server),
            scheduler,
            SocketWire.Server(lifetime, scheduler, port, "ksmt-server-socket"),
            lifetime
        )

        protocol.wire.connected.adviseForConditionAsync(lifetime).await()

        coroutineScope.launch(lifetime) {
            while (workerProcess.isAlive) {
                delay(checkProcessAliveDelay)
            }
            lifetime.terminate()
        }

        return RdServerProcess(workerProcess, lifetime, protocol, astSerializationCtx)
    }

    companion object{
        val logger = LoggerFactory.create(minLevel = LogLevel.Error)

        fun normalProcessFactory(entrypoint: KClass<*>, args: List<String>): ProcessWrapper =
            NormalProcess.start(entrypoint, args)

        fun processWithDebuggerFactory(entrypoint: KClass<*>, args: List<String>): ProcessWrapper =
            NormalProcess.startWithDebugger(entrypoint, args)
    }
}
