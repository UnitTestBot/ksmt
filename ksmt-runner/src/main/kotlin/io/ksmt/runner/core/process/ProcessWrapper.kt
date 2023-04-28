package io.ksmt.runner.core.process

interface ProcessWrapper {
    val isAlive: Boolean
    fun destroyForcibly()
}
