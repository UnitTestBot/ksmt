package io.ksmt.solver.z3

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.sort.KBoolSort

open class KZ3Solver(ctx: KContext) : KZ3SolverBase(ctx), KSolver<KZ3SolverConfiguration> {
    override val z3Ctx: KZ3Context = KZ3Context(ctx)
    private val trackedAssertions = ScopedArrayFrameOfLong2ObjectOpenHashMap<KExpr<KBoolSort>>()

    override fun saveTrackedAssertion(track: Long, trackedExpr: KExpr<KBoolSort>) {
        trackedAssertions.currentFrame[track] = trackedExpr
    }

    override fun findTrackedExprByTrack(track: Long): KExpr<KBoolSort>? = trackedAssertions.findNonNullValue {
        it[track]
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
