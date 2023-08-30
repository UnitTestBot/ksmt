package io.ksmt.solver.cvc5

import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import java.util.TreeMap
import java.util.TreeSet
import kotlin.time.Duration

open class KCvc5ForkingSolver internal constructor(
    ctx: KContext,
    private val manager: KCvc5ForkingSolverManager,
    parent: KCvc5ForkingSolver? = null
) : KCvc5SolverBase(ctx), KForkingSolver<KCvc5SolverConfiguration> {
    final override val cvc5Ctx: KCvc5ForkingContext = manager.createCvc5ForkingContext(solver, parent?.cvc5Ctx)
    private val isChild = parent != null
    private var assertionsInitiated = !isChild // don't need to initiate assertions for root solver

    private val trackedAssertions = ScopedLinkedFrame<TreeMap<Term, KExpr<KBoolSort>>>(::TreeMap, ::TreeMap)
    private val cvc5Assertions = ScopedLinkedFrame<TreeSet<Term>>(::TreeSet, ::TreeSet)

    override val currentScope: UInt
        get() = trackedAssertions.currentScope

    private val config: KCvc5ForkingSolverConfigurationImpl by lazy {
        parent?.config?.fork(solver) ?: KCvc5ForkingSolverConfigurationImpl(solver)
    }

    init {
        if (parent != null) {
            trackedAssertions.fork(parent.trackedAssertions)
            cvc5Assertions.fork(parent.cvc5Assertions)
            config // initialize child config
        }
    }

    override fun configure(configurator: KCvc5SolverConfiguration.() -> Unit) {
        config.configurator()
    }

    override fun fork(): KForkingSolver<KCvc5SolverConfiguration> = manager.mkForkingSolver(this)

    private fun ensureAssertionsInitiated() {
        if (assertionsInitiated) return

        cvc5Assertions.stacked()
            .zip(trackedAssertions.stacked())
            .asReversed()
            .forEachIndexed { scope, (cvc5AssertionFrame, trackedFrame) ->
                if (scope > 0) solver.push()

                cvc5AssertionFrame.forEach(solver::assertFormula)
                trackedFrame.forEach { (track, _) -> solver.assertFormula(track) }
            }

        assertionsInitiated = true
    }

    override fun saveTrackedAssertion(track: Term, trackedExpr: KExpr<KBoolSort>) {
        trackedAssertions.currentFrame[track] = trackedExpr
    }

    override fun findTrackedExprByTrack(track: Term) = trackedAssertions.findNonNullValue { it[track] }

    override fun assert(expr: KExpr<KBoolSort>): Unit = cvc5Ctx.cvc5Try {
        ctx.ensureContextMatch(expr)
        ensureAssertionsInitiated()

        val cvc5Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.assertFormula(cvc5Expr)
        cvc5Ctx.assertPendingAxioms(solver)
        cvc5Assertions.currentFrame.add(cvc5Expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        cvc5Ctx.cvc5Try { ensureAssertionsInitiated() }
        super.assertAndTrack(expr)
    }

    override fun push() {
        cvc5Ctx.cvc5Try { ensureAssertionsInitiated() }
        super.push()
        trackedAssertions.push()
        cvc5Assertions.push()
    }

    override fun pop(n: UInt) {
        cvc5Ctx.cvc5Try { ensureAssertionsInitiated() }
        super.pop(n)
        trackedAssertions.pop(n)
        cvc5Assertions.pop(n)
    }

    override fun check(timeout: Duration): KSolverStatus {
        cvc5Ctx.cvc5Try { ensureAssertionsInitiated() }
        cvc5Ctx.assertPendingAxioms(solver)
        return super.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        cvc5Ctx.cvc5Try { ensureAssertionsInitiated() }
        cvc5Ctx.assertPendingAxioms(solver)
        return super.checkWithAssumptions(assumptions, timeout)
    }

    override fun close() {
        manager.close(this)
        solver.close()
        cvc5Ctx.close()
    }
}
