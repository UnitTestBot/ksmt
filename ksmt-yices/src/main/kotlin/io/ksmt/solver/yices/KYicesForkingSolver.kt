package io.ksmt.solver.yices

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

class KYicesForkingSolver(
    ctx: KContext,
    private val manager: KYicesForkingSolverManager,
    parent: KYicesForkingSolver?,
) : KForkingSolver<KYicesSolverConfiguration>, KYicesSolverBase(ctx) {

    override val yicesCtx: KYicesForkingContext by lazy { KYicesForkingContext(ctx, manager, this) }

    private val trackedAssertions =
        ScopedLinkedFrame<ArrayList<Pair<KExpr<KBoolSort>, YicesTerm>>>(::ArrayList, ::ArrayList)
    private val yicesAssertions = ScopedLinkedFrame<HashSet<YicesTerm>>(::HashSet, ::HashSet)

    override val currentScope: UInt
        get() = trackedAssertions.currentScope

    private val ksmtConfig: KYicesForkingSolverConfigurationImpl by lazy {
        parent?.ksmtConfig?.fork(config) ?: KYicesForkingSolverConfigurationImpl(config)
    }

    private var assertionsInitiated = parent == null

    init {
        if (parent != null) {
            trackedAssertions.fork(parent.trackedAssertions)
            yicesAssertions.fork(parent.yicesAssertions)

            ksmtConfig // force initialization
        }
    }

    private fun ensureAssertionsInitiated() {
        if (assertionsInitiated) return

        yicesAssertions.stacked()
            .zip(trackedAssertions.stacked())
            .asReversed()
            .forEachIndexed { scope, (yicesAssertionFrame, _) ->
                if (scope > 0) nativeContext.push()

                yicesAssertionFrame.forEach(nativeContext::assertFormula)
            }

        assertionsInitiated = true
    }

    override fun configure(configurator: KYicesSolverConfiguration.() -> Unit) {
        requireActiveConfig()
        ksmtConfig.configurator()
    }

    /**
     * Creates lazily initiated forked solver with shared cache, preserving parental assertions and configuration.
     */
    override fun fork(): KForkingSolver<KYicesSolverConfiguration> = manager.mkForkingSolver(this)

    override fun saveTrackedAssertion(track: YicesTerm, trackedExpr: KExpr<KBoolSort>) {
        trackedAssertions.currentFrame += trackedExpr to track
    }

    override fun collectTrackedAssertions(collector: (Pair<KExpr<KBoolSort>, YicesTerm>) -> Unit) {
        trackedAssertions.forEach { frame ->
            frame.forEach(collector)
        }
    }

    override val hasTrackedAssertions: Boolean
        get() = trackedAssertions.any { it.isNotEmpty() }

    override fun assert(expr: KExpr<KBoolSort>) = yicesTry {
        yicesTry { ensureAssertionsInitiated() }
        ctx.ensureContextMatch(expr)

        val yicesExpr = with(exprInternalizer) { expr.internalize() }
        nativeContext.assertFormula(yicesExpr)
        yicesAssertions.currentFrame += yicesExpr
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        yicesTry { ensureAssertionsInitiated() }
        super.assertAndTrack(expr)
    }

    override fun push() {
        yicesTry { ensureAssertionsInitiated() }
        super.push()
        trackedAssertions.push()
        yicesAssertions.push()
    }

    override fun pop(n: UInt) {
        yicesTry { ensureAssertionsInitiated() }
        super.pop(n)
        trackedAssertions.pop(n)
        yicesAssertions.pop(n)
    }

    override fun check(timeout: Duration): KSolverStatus {
        yicesTry { ensureAssertionsInitiated() }
        return super.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        yicesTry { ensureAssertionsInitiated() }
        return super.checkWithAssumptions(assumptions, timeout)
    }

    override fun close() {
        super.close()
        manager.close(this)
    }
}
