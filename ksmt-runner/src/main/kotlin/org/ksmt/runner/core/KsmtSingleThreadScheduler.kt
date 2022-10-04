package org.ksmt.runner.core

import com.jetbrains.rd.util.threading.SingleThreadSchedulerBase

class KsmtSingleThreadScheduler(name: String) : SingleThreadSchedulerBase(name) {
    override fun onException(ex: Throwable) {
    }
}
