package org.ksmt.runner.core

import com.jetbrains.rd.framework.IProtocol
import com.jetbrains.rd.framework.IdKind
import com.jetbrains.rd.framework.Identities
import com.jetbrains.rd.framework.Protocol
import com.jetbrains.rd.framework.Serializers
import com.jetbrains.rd.framework.SocketWire
import com.jetbrains.rd.framework.base.static
import com.jetbrains.rd.framework.impl.RdCall
import com.jetbrains.rd.framework.impl.RdSignal
import com.jetbrains.rd.framework.util.launch
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.withTimeoutOrNull
import org.ksmt.runner.serializer.AstSerializationCtx
import kotlin.time.Duration

abstract class ChildProcessBase<Model> {

    private enum class State {
        STARTED,
        ENDED
    }

    private val synchronizer = Channel<State>(capacity = 1)

    abstract fun IProtocol.protocolModel(): Model
    abstract fun Model.setup(astSerializationCtx: AstSerializationCtx, onStop: () -> Unit)
    abstract fun parseArgs(args: Array<String>): KsmtWorkerArgs

    fun start(args: Array<String>) = runBlocking {
        val workerArgs = parseArgs(args)

        val def = LifetimeDefinition()
        def.launch {
            checkAliveLoop(def, workerArgs.workerTimeout)
        }

        def.usingNested { lifetime ->
            initiate(lifetime, workerArgs.port)
        }
    }

    private suspend fun checkAliveLoop(lifetime: LifetimeDefinition, timeout: Duration) {
        var lastState = State.STARTED
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
        val deferred = CompletableDeferred<Unit>()
        lifetime.onTermination { deferred.complete(Unit) }

        val scheduler = KsmtSingleThreadScheduler("ksmt-child-scheduler")
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
        val (sync, protocolModel) = obtainClientIO(lifetime, clientProtocol)

        protocolModel.setup(astSerializationCtx) {
            deferred.complete(Unit)
        }

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
        deferred.await()
    }

    private suspend fun obtainClientIO(lifetime: Lifetime, protocol: Protocol): Pair<RdSignal<String>, Model> =
        protocol.scheduler
            .pumpAsync(lifetime) {
                val sync = RdSignal<String>().static(1).apply {
                    async = true
                    bind(lifetime, protocol, rdid.toString())
                }
                sync to protocol.protocolModel()
            }
            .await()


    private fun <T> measureExecutionForTermination(block: () -> T): T = runBlocking {
        try {
            synchronizer.send(State.STARTED)
            return@runBlocking block()
        } finally {
            synchronizer.send(State.ENDED)
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
