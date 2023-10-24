package io.ksmt.solver.runner

import io.ksmt.expr.KExpr
import io.ksmt.runner.generated.models.SolverConfigurationParam
import io.ksmt.sort.KBoolSort
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
    private class AssertionFrame(
        val asserted: ConcurrentLinkedQueue<KExpr<KBoolSort>> = ConcurrentLinkedQueue(),
        val tracked: ConcurrentLinkedQueue<KExpr<KBoolSort>> = ConcurrentLinkedQueue(),
    )

    private val configuration = ConcurrentLinkedQueue<SolverConfigurationParam>()

    /**
     * Asserted expressions.
     * Each nested assertion frame contains expressions of the
     * corresponding assertion level.
     * */
    private val assertFrames = ConcurrentLinkedDeque<AssertionFrame>()

    init {
        assertFrames.addLast(AssertionFrame())
    }

    fun configure(config: List<SolverConfigurationParam>) {
        configuration.addAll(config)
    }

    fun assert(expr: KExpr<KBoolSort>) {
        assertFrames.last.asserted.add(expr)
    }

    fun assertAndTrack(expr: KExpr<KBoolSort>) {
        assertFrames.last.tracked.add(expr)
    }

    fun push() {
        assertFrames.addLast(AssertionFrame())
    }

    fun pop(n: UInt) {
        repeat(n.toInt()) {
            assertFrames.removeLast()
        }
    }

    suspend fun applyAsync(executor: KSolverRunnerExecutor) = replayState(
        configureSolver = { executor.configureAsync(it) },
        pushScope = { executor.pushAsync() },
        assertExprs = { executor.bulkAssertAsync(it) },
        assertExprsAndTrack = { expr -> executor.bulkAssertAndTrackAsync(expr) }
    )

    fun applySync(executor: KSolverRunnerExecutor) = replayState(
        configureSolver = { executor.configureSync(it) },
        pushScope = { executor.pushSync() },
        assertExprs = { executor.bulkAssertSync(it) },
        assertExprsAndTrack = { expr -> executor.bulkAssertAndTrackSync(expr) }
    )

    /**
     * Recover the solver state via re-applying
     * all solver state modification operations.
     * */
    private inline fun replayState(
        configureSolver: (List<SolverConfigurationParam>) -> Unit,
        pushScope: () -> Unit,
        assertExprs: (List<KExpr<KBoolSort>>) -> Unit,
        assertExprsAndTrack: (List<KExpr<KBoolSort>>) -> Unit
    ) {
        if (configuration.isNotEmpty()) {
            configureSolver(configuration.toList())
        }

        var firstFrame = true
        for (frame in assertFrames) {
            if (!firstFrame) {
                // Increase the solver assertion scope
                pushScope()
            }
            firstFrame = false

            assertExprs(frame.asserted.toList())
            assertExprsAndTrack(frame.tracked.toList())
        }
    }
}
