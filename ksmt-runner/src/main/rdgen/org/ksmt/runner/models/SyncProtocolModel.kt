package org.ksmt.runner.models

import com.jetbrains.rd.generator.nova.*

object SyncProtocolRoot: Root()

object SyncProtocolModel: Ext(SyncProtocolRoot) {
    init {
        signal("synchronizationSignal", PredefinedType.string).async
    }
}