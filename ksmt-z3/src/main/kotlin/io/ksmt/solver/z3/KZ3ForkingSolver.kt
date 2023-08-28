package io.ksmt.solver.z3

import com.microsoft.z3.Context
import com.microsoft.z3.solverAssert
import com.microsoft.z3.solverAssertAndTrack
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import it.unimi.dsi.fastutil.longs.LongOpenHashSet
import kotlin.time.Duration

open class KZ3ForkingSolver internal constructor(
    ctx: KContext,
    private val manager: KZ3ForkingSolverManager,
    parent: KZ3ForkingSolver?
) : KZ3SolverBase(ctx), KForkingSolver<KZ3SolverConfiguration> {
    final override val z3Ctx: KZ3Context

    private val trackedAssertions = ScopedLinkedFrame<Long2ObjectOpenHashMap<KExpr<KBoolSort>>>(
        ::Long2ObjectOpenHashMap, ::Long2ObjectOpenHashMap
    )
    private val z3Assertions = ScopedLinkedFrame(::LongOpenHashSet, ::LongOpenHashSet)

    private val isChild = parent != null
    private var assertionsInitiated = !isChild

    init {
        if (parent != null) {
            z3Ctx = parent.z3Ctx.fork(ctx, manager)
            trackedAssertions.fork(parent.trackedAssertions)
            z3Assertions.fork(parent.z3Assertions)
        } else {
            val context = Context()
            with(manager) { registerContext(context) }
            z3Ctx = KZ3Context(ctx, context, manager)
        }
    }

    private val config: KZ3ForkingSolverConfigurationImpl by lazy {
        z3Try {
            z3Ctx.nativeContext.mkParams().let {
                parent?.config?.fork(it)?.apply { setParameters(solver) } ?: KZ3ForkingSolverConfigurationImpl(it)
            }
        }
    }

    init {
        if (isChild) config // initialize child config
    }

    override fun configure(configurator: KZ3SolverConfiguration.() -> Unit) {
        config.configurator()
        config.setParameters(solver)
    }

    override fun fork(): KForkingSolver<KZ3SolverConfiguration> = manager.mkForkingSolver(this)

    override fun saveTrackedAssertion(track: Long, trackedExpr: KExpr<KBoolSort>) {
        trackedAssertions.currentFrame[track] = trackedExpr
    }

    override fun findTrackedExprByTrack(track: Long): KExpr<KBoolSort>? = trackedAssertions.findNonNullValue {
        it[track]
    }

    /**
     * Asserts parental (in case of child) assertions if not
     */
    private fun ensureAssertionsInitiated() {
        if (assertionsInitiated) return

        z3Assertions.stacked()
            .zip(trackedAssertions.stacked())
            .asReversed()
            .forEachIndexed { scope, (z3AssertionFrame, trackedFrame) ->
                if (scope > 0) {
                    solver.push()
                    currentScope++
                }

                z3AssertionFrame.forEach(solver::solverAssert)
                trackedFrame.forEach { (track, expr) ->
                    /** tracked [expr] was previously internalized by parent */
                    solver.solverAssertAndTrack(track, z3Ctx.findInternalizedExprWithoutAnalysis(expr))
                }
            }

        assertionsInitiated = true
    }

    override fun push() {
        z3Try { ensureAssertionsInitiated() }
        super.push()
        trackedAssertions.push()
        z3Assertions.push()
    }

    override fun pop(n: UInt) {
        z3Try { ensureAssertionsInitiated() }
        super.pop(n)
        trackedAssertions.pop(n)
        z3Assertions.pop(n)
    }

    override fun assert(expr: KExpr<KBoolSort>) = z3Try {
        ensureAssertionsInitiated()
        ctx.ensureContextMatch(expr)

        val z3Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.solverAssert(z3Expr)

        z3Ctx.assertPendingAxioms(solver)
        z3Assertions.currentFrame += z3Expr
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        z3Try { ensureAssertionsInitiated() }
        super.assertAndTrack(expr)
    }

    override fun check(timeout: Duration): KSolverStatus {
        z3Try { ensureAssertionsInitiated() }
        return super.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        z3Try { ensureAssertionsInitiated() }
        return super.checkWithAssumptions(assumptions, timeout)
    }
}
