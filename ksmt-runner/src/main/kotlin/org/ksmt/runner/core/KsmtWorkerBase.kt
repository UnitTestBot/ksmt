package org.ksmt.runner.core

import com.jetbrains.rd.framework.IProtocol
import kotlinx.coroutines.delay
import org.ksmt.runner.models.generated.syncProtocolModel
import kotlin.time.Duration.Companion.milliseconds

abstract class KsmtWorkerBase<Model>(
    val id: Int,
    private val rdServer: RdServer,
) : RdServer by rdServer {

    private var model: Model? = null

    val protocolModel: Model
        get() = model ?: error("Protocol model is not initialized")

    abstract fun initProtocolModel(protocol: IProtocol): Model

    suspend fun init(): KsmtWorkerBase<Model> {

        protocol.scheduler.pumpAsync(lifetime) {
            protocol.syncProtocolModel
            model = initProtocolModel(protocol)
        }.await()


        protocol.syncProtocolModel.synchronizationSignal.let { sync ->
            val messageFromChild = sync.adviseForConditionAsync(lifetime) {
                it == CHILD_PROCESS_NAME
            }

            while (messageFromChild.isActive) {
                sync.fire(MAIN_PROCESS_NAME)
                delay(20.milliseconds)
            }
        }

        return this
    }
}
