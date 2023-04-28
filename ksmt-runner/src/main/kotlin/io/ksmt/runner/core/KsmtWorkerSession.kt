package io.ksmt.runner.core

import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.lifetime.isAlive

class KsmtWorkerSession<Model>(
    override val lifetime: LifetimeDefinition,
    private val worker: KsmtWorkerBase<Model>,
    private val pool: KsmtWorkerPool<Model>
) : RdServer by worker, Lifetimed {
    override val isAlive: Boolean
        get() = lifetime.isAlive && worker.isAlive

    val protocolModel: Model
        get() = worker.protocolModel

    fun release() {
        lifetime.terminate()
        pool.releaseWorker(worker)
    }

    override fun terminate() {
        lifetime.terminate()
        pool.killWorker(worker)
    }
}
