package org.ksmt.runner.core

import com.jetbrains.rd.framework.IProtocol
import com.jetbrains.rd.util.lifetime.Lifetime
import org.ksmt.runner.serializer.AstSerializationCtx

interface RdServer {
    val isAlive: Boolean
    val lifetime: Lifetime
    val protocol: IProtocol
    val astSerializationCtx: AstSerializationCtx
    fun terminate()
}
