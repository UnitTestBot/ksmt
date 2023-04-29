package io.ksmt.runner.core

import kotlin.reflect.KClass

interface KsmtWorkerFactory<Model> {
    val childProcessEntrypoint: KClass<out ChildProcessBase<*>>
    fun mkWorker(id: Int, process: RdServer): KsmtWorkerBase<Model>
    fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs
}
