package io.ksmt.solver.cvc5

import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.sort.KBoolSort
import java.util.TreeMap

open class KCvc5Solver(ctx: KContext) : KCvc5SolverBase(ctx), KSolver<KCvc5SolverConfiguration> {
    override val cvc5Ctx: KCvc5Context = KCvc5Context(solver, ctx)
    private val trackedAssertions = ScopedArrayFrame<TreeMap<Term, KExpr<KBoolSort>>> { TreeMap() }

    override val currentScope: UInt
        get() = trackedAssertions.currentScope

    override fun saveTrackedAssertion(track: Term, trackedExpr: KExpr<KBoolSort>) {
        trackedAssertions.currentFrame[track] = trackedExpr
    }

    override fun findTrackedExprByTrack(track: Term) = trackedAssertions.findNonNullValue { it[track] }

    override fun push() {
        super.push()
        trackedAssertions.push()
    }

    override fun pop(n: UInt) {
        super.pop(n)
        trackedAssertions.pop(n)
    }
}
