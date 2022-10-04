package org.ksmt.runner.core

import com.jetbrains.rd.framework.base.static
import com.jetbrains.rd.framework.impl.RdSignal
import kotlinx.coroutines.delay
import kotlin.time.Duration

abstract class KsmtWorkerBase<Model>(
    val id: Int,
    private val rdServer: RdServer
) : RdServer by rdServer {
    private val sync = RdSignal<String>().static(1).apply { async = true }

    abstract val protocolModel: Model

    suspend fun init(keepAliveMessagePeriod: Duration): KsmtWorkerBase<Model> {

        protocol.scheduler.pumpAsync(lifetime) {
            sync.bind(lifetime, protocol, sync.rdid.toString())
            protocolModel
        }.await()


        val messageFromChild = sync.adviseForConditionAsync(lifetime) {
            it == CHILD_PROCESS_NAME
        }

        while (messageFromChild.isActive) {
            sync.fire(MAIN_PROCESS_NAME)
            delay(keepAliveMessagePeriod)
        }

        return this
    }
}
