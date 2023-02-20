package org.ksmt.solver.runner

import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.runner.generated.models.SolverConfigurationParam
import org.ksmt.sort.KBoolSort
import java.util.concurrent.ConcurrentLinkedDeque
import java.util.concurrent.ConcurrentLinkedQueue

/**
 * State of an SMT solver.
 * Allows us to recover the state of the solver after failures.
 *
 * The state of the solver:
 * 1. Solver configuration
 * 2. Asserted expressions
 * 3. Assertion level (push commands)
 *
 * We don't consider last check-sat result as a solver state.
 * */
class KSolverState {
    private sealed interface AssertFrame
    private data class ExprAssertFrame(val expr: KExpr<KBoolSort>) : AssertFrame
    private data class AssertAndTrackFrame(
        val expr: KExpr<KBoolSort>,
        val trackVar: KConstDecl<KBoolSort>
    ) : AssertFrame

    private val configuration = ConcurrentLinkedQueue<SolverConfigurationParam>()

    /**
     * Asserted expressions.
     * Each nested queue contains expressions of the
     * corresponding assertion level.
     * */
    private val assertFrames = ConcurrentLinkedDeque<ConcurrentLinkedQueue<AssertFrame>>()

    init {
        assertFrames.addLast(ConcurrentLinkedQueue())
    }

    fun configure(config: List<SolverConfigurationParam>) {
        configuration.addAll(config)
    }

    fun assert(expr: KExpr<KBoolSort>) {
        assertFrames.last.add(ExprAssertFrame(expr))
    }

    fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        assertFrames.last.add(AssertAndTrackFrame(expr, trackVar))
    }

    fun push() {
        assertFrames.addLast(ConcurrentLinkedQueue())
    }

    fun pop(n: UInt) {
        repeat(n.toInt()) {
            assertFrames.removeLast()
        }
    }

    /**
     * Recover the solver state via re-applying
     * all solver state modification operations.
     * */
    suspend fun apply(executor: KSolverRunnerExecutor) {
        if (configuration.isNotEmpty()) {
            executor.configureAsync(configuration.toList())
        }

        var firstFrame = true
        for (frame in assertFrames) {
            if (!firstFrame) {
                // Increase the solver assertion scope
                executor.pushAsync()
            }
            firstFrame = false

            for (assertion in frame) {
                when (assertion) {
                    is ExprAssertFrame -> executor.assertAsync(assertion.expr)
                    is AssertAndTrackFrame -> executor.assertAndTrackAsync(assertion.expr, assertion.trackVar)
                }
            }
        }
    }
}
