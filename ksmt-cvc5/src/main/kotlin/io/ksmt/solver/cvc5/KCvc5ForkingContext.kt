package io.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort

class KCvc5ForkingContext private constructor(
    solver: Solver,
    mkExprSolver: Solver,
    ctx: KContext,
    manager: KCvc5ForkingSolverManager,
    parent: KCvc5ForkingContext?
) : KCvc5Context(solver, mkExprSolver, ctx) {
    constructor(solver: Solver, mkExprSolver: Solver, ctx: KContext, manager: KCvc5ForkingSolverManager) : this(
        solver, mkExprSolver, ctx, manager, null
    )

    private val uninterpretedSortsLinkedFrame = ScopedLinkedFrame<HashSet<KUninterpretedSort>>(::HashSet, ::HashSet)
    private val declarationsLinkedFrame = ScopedLinkedFrame<HashSet<KDecl<*>>>(::HashSet, ::HashSet)

    override val uninterpretedSorts: ScopedFrame<HashSet<KUninterpretedSort>>
        get() = uninterpretedSortsLinkedFrame
    override val declarations: ScopedFrame<HashSet<KDecl<*>>>
        get() = declarationsLinkedFrame

    override val expressions = with(manager) { getExpressionsCache() }
    override val cvc5Expressions = with(manager) { getExpressionsReversedCache() }
    override val sorts = with(manager) { getSortsCache() }
    override val cvc5Sorts = with(manager) { getSortsReversedCache() }
    override val decls = with(manager) { getDeclsCache() }
    override val cvc5Decls = with(manager) { getDeclsReversedCache() }

    override val uninterpretedSortValueInterpreter = with(manager) { getUninterpretedSortsValueInterpretersCache() }

    /**
     * Uninterpreted sort values and universe are shared for whole forking hierarchy (from parent to children)
     * due to shared expressions cache,
     * that's why once [registerUninterpretedSortValue] and [saveUninterpretedSortValue] are called,
     * each solver in hierarchy should assert newly internalized uninterpreted sort values via [assertPendingAxioms]
     *
     * @see KCvc5Model.uninterpretedSortUniverse
     */
    override val uninterpretedSortValues = with(manager) { getUninterpretedSortValues() }

    override val uninterpretedValuesTracker: ExpressionUninterpretedValuesForkingTracker = parent
        ?.uninterpretedValuesTracker?.fork(this)
        ?: ExpressionUninterpretedValuesForkingTracker(this)


    init {
        if (parent != null) {
            currentAccumulatedScopeExpressions += parent.currentAccumulatedScopeExpressions
            uninterpretedSortsLinkedFrame.fork(parent.uninterpretedSortsLinkedFrame)
            declarationsLinkedFrame.fork(parent.declarationsLinkedFrame)
        }
    }

    fun fork(solver: Solver, forkingSolverManager: KCvc5ForkingSolverManager): KCvc5ForkingContext {
        ensureActive()
        return KCvc5ForkingContext(solver, mkExprSolver, ctx, forkingSolverManager, this)
    }

    override fun close() {
        if (isClosed) return
        isClosed = true
    }

    override fun <T : KSort> Term.convert(converter: KCvc5ExprConverter): KExpr<T> = with(converter) {
        convertExprWithMkExprSolver()
    }
}
