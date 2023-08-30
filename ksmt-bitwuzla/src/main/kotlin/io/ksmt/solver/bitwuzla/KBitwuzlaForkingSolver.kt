package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

class KBitwuzlaForkingSolver(
    private val ctx: KContext,
    private val manager: KBitwuzlaForkingSolverManager,
    parent: KBitwuzlaForkingSolver?
) : KBitwuzlaSolverBase(ctx),
    KForkingSolver<KBitwuzlaSolverConfiguration> {

    private val assertions = ScopedLinkedFrame<MutableList<KExpr<KBoolSort>>>(::ArrayList, ::ArrayList)
    private val trackToExprFrames =
        ScopedLinkedFrame<MutableList<Pair<KExpr<KBoolSort>, KExpr<KBoolSort>>>>(::ArrayList, ::ArrayList)

    private val config: KBitwuzlaForkingSolverConfigurationImpl

    init {
        if (parent != null) {
            config = parent.config.fork(bitwuzlaCtx.bitwuzla)
            assertions.fork(parent.assertions)
            trackToExprFrames.fork(parent.trackToExprFrames)
        } else {
            config = KBitwuzlaForkingSolverConfigurationImpl(bitwuzlaCtx.bitwuzla)
        }
    }

    override fun configure(configurator: KBitwuzlaSolverConfiguration.() -> Unit) {
        config.configurator()
    }

    /**
     * Creates lazily initiated forked solver (without cache sharing), preserving parental assertions and configuration.
     */
    override fun fork(): KForkingSolver<KBitwuzlaSolverConfiguration> = manager.createForkingSolver(this)

    private var assertionsInitiated = parent == null

    private fun ensureAssertionsInitiated() {
        if (assertionsInitiated) return

        assertions.stacked().zip(trackToExprFrames.stacked())
            .asReversed()
            .forEachIndexed { scope, (assertionsFrame, trackedExprsFrame) ->
                if (scope > 0) super.push()

                assertionsFrame.forEach { assertion ->
                    internalizeAndAssertWithAxioms(assertion)
                }

                trackedExprsFrame.forEach { (track, trackedExpr) ->
                    super.registerTrackForExpr(trackedExpr, track)
                }
            }
        assertionsInitiated = true
    }

    override fun assert(expr: KExpr<KBoolSort>) = bitwuzlaCtx.bitwuzlaTry {
        ctx.ensureContextMatch(expr)
        ensureAssertionsInitiated()

        internalizeAndAssertWithAxioms(expr)
        assertions.currentFrame += expr
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        bitwuzlaCtx.bitwuzlaTry { ensureAssertionsInitiated() }
        super.assertAndTrack(expr)
    }

    override fun registerTrackForExpr(expr: KExpr<KBoolSort>, track: KExpr<KBoolSort>) {
        super.registerTrackForExpr(expr, track)
        trackToExprFrames.currentFrame += track to expr
    }

    override fun push() {
        bitwuzlaCtx.bitwuzlaTry { ensureAssertionsInitiated() }
        super.push()
        assertions.push()
        trackToExprFrames.push()
    }

    override fun pop(n: UInt) {
        bitwuzlaCtx.bitwuzlaTry { ensureAssertionsInitiated() }
        super.pop(n)
        assertions.pop(n)
        trackToExprFrames.pop(n)
    }

    override fun check(timeout: Duration): KSolverStatus {
        bitwuzlaCtx.bitwuzlaTry { ensureAssertionsInitiated() }
        return super.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        bitwuzlaCtx.bitwuzlaTry { ensureAssertionsInitiated() }
        return super.checkWithAssumptions(assumptions, timeout)
    }

    override fun close() {
        super.close()
        manager.close(this)
    }
}
