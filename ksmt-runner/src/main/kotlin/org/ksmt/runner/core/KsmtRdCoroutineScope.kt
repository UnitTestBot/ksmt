package org.ksmt.runner.core

import com.jetbrains.rd.framework.util.RdCoroutineScope
import com.jetbrains.rd.framework.util.asCoroutineDispatcher
import com.jetbrains.rd.util.lifetime.Lifetime

class KsmtRdCoroutineScope(
    lifetime: Lifetime,
    scheduler: KsmtSingleThreadScheduler
) : RdCoroutineScope(lifetime) {
    override val defaultDispatcher = scheduler.asCoroutineDispatcher
}
