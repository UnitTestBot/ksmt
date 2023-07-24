package io.ksmt.solver.cvc5

import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import java.util.TreeMap
import java.util.TreeSet
import kotlin.time.Duration

open class KCvc5ForkingSolver internal constructor(
    ctx: KContext,
    private val manager: KCvc5ForkingSolverManager,
    parent: KCvc5ForkingSolver?
) : KCvc5SolverBase(ctx), KForkingSolver<KCvc5SolverConfiguration>, KSolver<KCvc5SolverConfiguration> {

    final override val cvc5Ctx: KCvc5Context
    private val isChild = parent != null
    private var assertionsInitiated = !isChild

    private val _trackedAssertions: ScopedLinkedFrame<TreeMap<Term, KExpr<KBoolSort>>>

    override val trackedAssertions: ScopedFrame<TreeMap<Term, KExpr<KBoolSort>>>
        get() = _trackedAssertions

    private val cvc5Assertions: ScopedLinkedFrame<TreeSet<Term>>

    init {
        if (parent != null) {
            cvc5Ctx = parent.cvc5Ctx.fork(solver)
            _trackedAssertions = parent._trackedAssertions.fork()
            cvc5Assertions = parent.cvc5Assertions.fork()
        } else {
            cvc5Ctx = KCvc5Context(solver, ctx, true)
            _trackedAssertions = ScopedLinkedFrame(::TreeMap, ::TreeMap)
            cvc5Assertions = ScopedLinkedFrame(::TreeSet, ::TreeSet)
        }
    }

    private val config: KCvc5ForkingSolverConfigurationImpl by lazy {
        parent?.config?.fork(solver) ?: KCvc5ForkingSolverConfigurationImpl(solver)
    }

    override fun configure(configurator: KCvc5SolverConfiguration.() -> Unit) {
        config.configurator()
    }

    override fun fork(): KForkingSolver<KCvc5SolverConfiguration> = manager.mkForkingSolver(this)

    private fun ensureAssertionsInitiated() {
        if (assertionsInitiated) return

        cvc5Assertions.stacked()
            .zip(_trackedAssertions.stacked())
            .asReversed()
            .forEachIndexed { scope, (cvc5AssertionFrame, trackedFrame) ->
                if (scope > 0) solver.push()

                cvc5AssertionFrame.forEach(solver::assertFormula)
                trackedFrame.forEach { (track, _) -> solver.assertFormula(track) }
            }

        assertionsInitiated = true
    }

    override fun assert(expr: KExpr<KBoolSort>): Unit = cvc5Try {
        ctx.ensureContextMatch(expr)
        ensureAssertionsInitiated()

        val cvc5Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.assertFormula(cvc5Expr)
        cvc5Ctx.assertPendingAxioms(solver)
        cvc5Assertions.currentFrame.add(cvc5Expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        cvc5Try { ensureAssertionsInitiated() }
        super.assertAndTrack(expr)
    }

    override fun push() {
        cvc5Try { ensureAssertionsInitiated() }
        super.push()
        cvc5Assertions.push()
    }

    override fun pop(n: UInt) {
        cvc5Try { ensureAssertionsInitiated() }
        super.pop(n)
        cvc5Assertions.pop(n)
    }

    override fun check(timeout: Duration): KSolverStatus {
        cvc5Try { ensureAssertionsInitiated() }
        cvc5Ctx.assertPendingAxioms(solver)
        return super.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        cvc5Try { ensureAssertionsInitiated() }
        cvc5Ctx.assertPendingAxioms(solver)
        return super.checkWithAssumptions(assumptions, timeout)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        val cvc5FullCore = cvc5UnsatCore()

        val unsatCore = mutableListOf<KExpr<KBoolSort>>()

        cvc5FullCore.forEach { unsatCoreTerm ->
            lastCvc5Assumptions?.get(unsatCoreTerm)?.also { unsatCore += it }
                ?: trackedAssertions.find { trackedAssertion ->
                    trackedAssertion[unsatCoreTerm]?.let { unsatCore += it; true } ?: false
                }
        }
        return unsatCore
    }

    override fun close() {
        manager.close(this)
        super.close()
    }
}
