package io.ksmt.solver.cvc5

import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolver
import io.ksmt.sort.KBoolSort
import java.util.TreeMap

open class KCvc5Solver(ctx: KContext) : KCvc5SolverBase(ctx), KSolver<KCvc5SolverConfiguration> {

    override val cvc5Ctx: KCvc5Context = KCvc5Context(solver, ctx)
    override val trackedAssertions: ScopedFrame<TreeMap<Term, KExpr<KBoolSort>>> = ScopedArrayFrame { TreeMap() }
}
