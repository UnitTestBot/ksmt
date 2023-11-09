package io.ksmt.solver.maxsmt.solvers.utils

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.KSolverStatus.UNKNOWN
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.utils.CoreUtils
import io.ksmt.solver.maxsmt.utils.ModelUtils
import io.ksmt.solver.maxsmt.utils.TimerUtils
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

internal class MinimalUnsatCore<T : KSolverConfiguration>(
    private val ctx: KContext,
    private val solver: KSolver<T>,
) {
    private val _minimalUnsatCoreModel = MinimalUnsatCoreModel(ctx, solver)

    fun getBestModel(): Pair<KModel?, UInt> = _minimalUnsatCoreModel.getBestModel()

    // If solver starts returning unknown, we return non minimized unsat core.
    fun tryGetMinimalUnsatCore(
        assumptions: List<SoftConstraint>,
        timeout: Duration = Duration.INFINITE,
    ): List<SoftConstraint> = with(ctx) {
        val clockStart = System.currentTimeMillis()

        val unsatCore = solver.unsatCore()

        if (unsatCore.isEmpty()) {
            return CoreUtils.coreToSoftConstraints(unsatCore, assumptions)
        }

        val minimalUnsatCore = mutableListOf<KExpr<KBoolSort>>()

        val unknown = unsatCore.toMutableList()

        while (unknown.isNotEmpty()) {
            val remainingTime = TimerUtils.computeRemainingTime(timeout, clockStart)
            if (TimerUtils.timeoutExceeded(remainingTime)) {
                return CoreUtils.coreToSoftConstraints(unsatCore, assumptions)
            }

            val expr = unknown.removeLast()

            val notExpr = !expr
            minimalUnsatCore.add(notExpr)

            val status = solver.checkWithAssumptions(minimalUnsatCore + unknown, remainingTime)

            when (status) {
                UNKNOWN -> return CoreUtils.coreToSoftConstraints(unsatCore, assumptions)

                SAT -> {
                    minimalUnsatCore.add(expr)
                    _minimalUnsatCoreModel.updateModel(assumptions)
                }

                else -> processUnsat(notExpr, unknown, minimalUnsatCore)
            }
        }

        return CoreUtils.coreToSoftConstraints(unsatCore, assumptions)
    }

    fun reset() = _minimalUnsatCoreModel.reset()

    private fun processUnsat(
        notExpr: KExpr<KBoolSort>,
        unknown: MutableList<KExpr<KBoolSort>>,
        minimalUnsatCore: List<KExpr<KBoolSort>>,
    ) {
        val core = solver.unsatCore()

        if (!core.contains(notExpr)) {
            // unknown := core \ mus
            unknown.clear()

            for (e in core) {
                if (!minimalUnsatCore.contains(e)) {
                    unknown.add(e)
                }
            }
        }
    }

    private class MinimalUnsatCoreModel<T : KSolverConfiguration>(
        private val ctx: KContext,
        private val solver: KSolver<T>,
    ) {
        private var _model: KModel? = null
        private var _weight = 0u

        fun getBestModel(): Pair<KModel?, UInt> {
            return Pair(_model, _weight)
        }

        fun updateModel(assumptions: List<SoftConstraint>) {
            reset()

            if (assumptions.isEmpty()) {
                return
            }

            val model = solver.model().detach()
            var weight = 0u

            for (asm in assumptions) {
                if (ModelUtils.expressionIsFalse(ctx, model, asm.expression)) {
                    weight += asm.weight
                }
            }

            if (_model == null || weight < _weight) {
                _model = model
                _weight = weight
            }
        }

        fun reset() {
            _model = null
            _weight = 0u
        }
    }
}
