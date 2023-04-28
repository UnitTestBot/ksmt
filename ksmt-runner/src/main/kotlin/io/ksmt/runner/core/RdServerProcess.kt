package io.ksmt.runner.core

import com.jetbrains.rd.framework.Protocol
import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.lifetime.isAlive
import io.ksmt.runner.core.process.ProcessWrapper
import io.ksmt.runner.serializer.AstSerializationCtx

class RdServerProcess(
    private val process: ProcessWrapper,
    override val lifetime: LifetimeDefinition,
    override val protocol: Protocol,
    override val astSerializationCtx: AstSerializationCtx
) : RdServer {
    override val isAlive: Boolean
        get() = lifetime.isAlive && process.isAlive

    override fun terminate() {
        lifetime.terminate()
    }
}
