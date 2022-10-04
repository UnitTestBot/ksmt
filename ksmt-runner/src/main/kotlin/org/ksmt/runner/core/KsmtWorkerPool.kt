package org.ksmt.runner.core

import com.jetbrains.rd.framework.IdKind
import com.jetbrains.rd.framework.Identities
import com.jetbrains.rd.framework.Protocol
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.SocketWire
import com.jetbrains.rd.framework.util.NetUtils
import com.jetbrains.rd.framework.util.synchronizeWith
import com.jetbrains.rd.util.AtomicInteger
import com.jetbrains.rd.util.concurrentMapOf
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.lifetime.isAlive
import com.jetbrains.rd.util.lifetime.throwIfNotAlive
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.coroutines.withTimeout
import kotlinx.coroutines.withTimeoutOrNull
import org.ksmt.runner.serializer.AstSerializationCtx
import kotlin.reflect.KClass
import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds
import kotlin.time.Duration.Companion.seconds

class KsmtWorkerPool<Model>(
    private val maxWorkerPoolSize: Int = 1,
    private val initializationTimeout: Duration = 5.seconds,
    private val checkProcessAliveDelay: Duration = 1.seconds,
    private val workerProcessIdleTimeout: Duration = 10.seconds,
    private val keepAliveMessagePeriod: Duration = 10.milliseconds,
    private val workerFactory: KsmtWorkerFactory<Model>
) : AutoCloseable {
    private val readyWorkers = Channel<KsmtWorkerBase<Model>?>(capacity = Channel.UNLIMITED)
    private val workers = concurrentMapOf<Int, KsmtWorkerBase<Model>>()
    private val activeWorkers = AtomicInteger(0)
    private val workerId = AtomicInteger(1)
    private val ldef = LifetimeDefinition()
    private val scheduler = KsmtSingleThreadScheduler("ksmt-runner-scheduler")
    private val coroutineScope = KsmtRdCoroutineScope(ldef, scheduler)

    suspend fun getOrCreateFreeWorker(): KsmtWorkerBase<Model> {
        while (ldef.isAlive) {
            if (activeWorkers.get() < maxWorkerPoolSize) {
                createWorker()
                continue
            }
            val worker = withTimeoutOrNull(initializationTimeout) {
                readyWorkers.receive()
            }
            if (worker != null && checkWorker(worker)) {
                return worker
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
        if (workers.remove(worker.id) == null) return
        activeWorkers.decrementAndGet()

        worker.terminate()

        // awake all waiting clients in getOrCreateFreeWorker
        readyWorkers.trySendBlocking(null)
    }

    override fun close() {
        ldef.terminate()
        readyWorkers.close()
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
        val workerLifetime = ldef.createNested()
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
            val rdProcess = startProcessWithRdServer(lifetime) { port ->
                val basicArgs = KsmtWorkerArgs(id, port, workerProcessIdleTimeout)
                val args = workerFactory.updateArgs(basicArgs)

                startProcess(workerFactory.childProcessEntrypoint, args.toList())
            }
            val worker = workerFactory.mkWorker(id, rdProcess)
            worker.init(keepAliveMessagePeriod)
            worker
        }

    private suspend fun startProcessWithRdServer(
        lifetime: Lifetime,
        factory: (Int) -> Process
    ): RdServerProcess {
        lifetime.throwIfNotAlive()

        val port = NetUtils.findFreePort(0)

        val worker = factory(port)

        return initRdServer(lifetime.createNested(), worker, port)
    }

    private suspend fun initRdServer(lifetime: LifetimeDefinition, workerProcess: Process, port: Int): RdServerProcess {
        lifetime.onTermination {
            workerProcess.destroyForcibly()
        }
        val nestedDef = lifetime.createNested()
        val aliveCheckJob = coroutineScope.launch {
            while (workerProcess.isAlive) {
                delay(checkProcessAliveDelay)
            }
            lifetime.terminate()
        }
        nestedDef.synchronizeWith(aliveCheckJob)

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

        return RdServerProcess(workerProcess, lifetime, protocol, astSerializationCtx)
    }

    private fun startProcess(entrypoint: KClass<*>, args: List<String>): Process {
        val classPath = System.getProperty("java.class.path") ?: error("No class path")
        val entrypointClassName = entrypoint.qualifiedName ?: error("Entrypoint class name is not available")
        val javaExecutable = ProcessHandle.current().info().command().orElseThrow()
        val workerCommand = listOf(
            javaExecutable,
            "-classpath", classPath,
        ) + listOf(
            entrypointClassName
        ) + args
        val pb = ProcessBuilder(workerCommand).inheritIO()
        return pb.start()
    }

}
