package org.ksmt.runner.core

import com.jetbrains.rd.framework.IProtocol
import org.ksmt.runner.serializer.AstSerializationCtx

interface RdServer: Lifetimed {
    val isAlive: Boolean
    val protocol: IProtocol
    val astSerializationCtx: AstSerializationCtx
}
