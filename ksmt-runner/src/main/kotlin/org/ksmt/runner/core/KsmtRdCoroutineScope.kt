package org.ksmt.runner.core

import com.jetbrains.rd.framework.util.RdCoroutineScope
import com.jetbrains.rd.framework.util.asCoroutineDispatcher
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.threading.SingleThreadScheduler

class KsmtRdCoroutineScope(
    lifetime: Lifetime,
    scheduler: SingleThreadScheduler
) : RdCoroutineScope(lifetime) {
    override val defaultDispatcher = scheduler.asCoroutineDispatcher
}
