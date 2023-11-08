package io.ksmt.runner.core

import com.jetbrains.rd.framework.IProtocol
import com.jetbrains.rd.framework.IdKind
import com.jetbrains.rd.framework.Identities
import com.jetbrains.rd.framework.Protocol
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.SocketWire
import com.jetbrains.rd.framework.impl.RdCall
import com.jetbrains.rd.framework.util.launch
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.threading.SingleThreadScheduler
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.channels.trySendBlocking
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import io.ksmt.runner.generated.models.syncProtocolModel
import io.ksmt.runner.serializer.AstSerializationCtx
import kotlin.time.Duration

abstract class ChildProcessBase<Model> {

    private enum class State {
        STARTED,
        ENDED
    }

    private val synchronizer = Channel<State>(capacity = 1)

    abstract fun initProtocolModel(protocol: IProtocol): Model
    abstract fun Model.setup(astSerializationCtx: AstSerializationCtx, lifetime: Lifetime)
    abstract fun parseArgs(args: Array<String>): KsmtWorkerArgs

    fun start(args: Array<String>) = runBlocking {
        val workerArgs = parseArgs(args)
        LoggerFactory.create(workerArgs.logLevel)

        val def = LifetimeDefinition()
        def.terminateOnException {
            def.launch {
                checkAliveLoop(def, workerArgs.workerTimeout)
            }

            initiate(def, workerArgs.port)

            def.awaitTermination()
        }
    }

    private suspend fun checkAliveLoop(lifetime: LifetimeDefinition, timeout: Duration) {
        var lastState = State.ENDED
        while (true) {
            val current = withTimeoutOrNull(timeout) {
                synchronizer.receive()
            }

            if (current == null) {
                if (lastState == State.ENDED) {
                    lifetime.terminate()
                    break
                }
            } else {
                lastState = current
            }
        }
    }

    private suspend fun initiate(lifetime: Lifetime, port: Int) {
        val scheduler = SingleThreadScheduler(lifetime, "ksmt-child-scheduler")
        val serializers = Serializers()
        val astSerializationCtx = AstSerializationCtx.register(serializers)
        val clientProtocol = Protocol(
            "ksmt-child",
            serializers,
            Identities(IdKind.Client),
            scheduler,
            SocketWire.Client(lifetime, scheduler, port),
            lifetime
        )

        val protocolModel = clientProtocol.scheduler.pumpAsync(lifetime) {
            clientProtocol.syncProtocolModel
            initProtocolModel(clientProtocol)
        }.await()

        protocolModel.setup(astSerializationCtx, lifetime)

        clientProtocol.syncProtocolModel.synchronizationSignal.let { sync ->
            val answerFromMainProcess = sync.adviseForConditionAsync(lifetime) {
                if (it == MAIN_PROCESS_NAME) {
                    measureExecutionForTermination {
                        sync.fire(CHILD_PROCESS_NAME)
                    }
                    true
                } else {
                    false
                }
            }
            answerFromMainProcess.await()
        }
    }

    private inline fun <T> measureExecutionForTermination(block: () -> T): T {
        try {
            synchronizer.trySendBlocking(State.STARTED).exceptionOrNull()
            return block()
        } finally {
            synchronizer.trySendBlocking(State.ENDED).exceptionOrNull()
        }
    }

    fun <T, R> RdCall<T, R>.measureExecutionForTermination(block: (T) -> R) {
        set { request ->
            measureExecutionForTermination<R> {
                block(request)
            }
        }
    }
}
