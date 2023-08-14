package io.ksmt.solver.yices

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class KYicesSolver(ctx: KContext) : KYicesSolverBase(ctx) {
    override val yicesCtx = KYicesContext(ctx)

    private val trackedAssertions = ScopedArrayFrame<ArrayList<Pair<KExpr<KBoolSort>, YicesTerm>>>(::ArrayList)
    override val currentScope: UInt
        get() = trackedAssertions.currentScope

    override val hasTrackedAssertions: Boolean
        get() = trackedAssertions.any { it.isNotEmpty() }

    override fun saveTrackedAssertion(track: YicesTerm, trackedExpr: KExpr<KBoolSort>) {
        trackedAssertions.currentFrame += trackedExpr to track
    }

    override fun collectTrackedAssertions(collector: (Pair<KExpr<KBoolSort>, YicesTerm>) -> Unit) {
        trackedAssertions.forEach { frame ->
            frame.forEach(collector)
        }
    }

    override fun configure(configurator: KYicesSolverConfiguration.() -> Unit) {
        requireActiveConfig()
        KYicesSolverConfigurationImpl(config).configurator()
    }

    override fun push() {
        super.push()
        trackedAssertions.push()
    }

    override fun pop(n: UInt) {
        super.pop(n)
        trackedAssertions.pop(n)
    }
}
