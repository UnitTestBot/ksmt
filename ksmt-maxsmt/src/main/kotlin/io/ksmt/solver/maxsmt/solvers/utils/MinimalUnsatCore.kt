package io.ksmt.solver.maxsmt.solvers.utils

import io.github.oshai.kotlinlogging.KotlinLogging
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
import kotlin.time.TimeSource.Monotonic.markNow

internal class MinimalUnsatCore<T : KSolverConfiguration>(
    private val ctx: KContext,
    private val solver: KSolver<T>,
) {
    private val _minimalUnsatCoreModel = MinimalUnsatCoreModel(ctx, solver)
    private lateinit var coreStatistics: MinimalCoreStatistics
    private val logger = KotlinLogging.logger {}

    fun getBestModel(): Pair<KModel?, UInt> = _minimalUnsatCoreModel.getBestModel()

    // If solver starts returning unknown or exceeds the timeout, non-minimized unsat core is returned.
    fun tryGetMinimalUnsatCore(
        assumptions: List<SoftConstraint>,
        timeout: Duration = Duration.INFINITE,
        collectStatistics: Boolean = false,
    ): List<SoftConstraint> = with(ctx) {
        logger.info { "core minimization --- started" }

        val markStart = markNow()

        if (collectStatistics) {
            coreStatistics = MinimalCoreStatistics()
        }

        val unsatCore = solver.unsatCore()

        if (unsatCore.isEmpty()) {
            logger.info { "core minimization ended --- unsat core is empty" }
            return emptyList()
        }

        val minimalUnsatCore = mutableListOf<KExpr<KBoolSort>>()

        val unknown = unsatCore.toMutableList()

        while (unknown.isNotEmpty()) {
            val remainingTime = TimerUtils.computeRemainingTime(timeout, markStart)
            if (TimerUtils.timeoutExceeded(remainingTime)) {
                logger.info { "core minimization ended --- timeout exceeded" }
                return CoreUtils.coreToSoftConstraints(unsatCore, assumptions)
            }

            val expr = unknown.removeLast()

            val notExpr = !expr

            val markCheckStart = markNow()
            val status = solver.checkWithAssumptions(minimalUnsatCore + notExpr + unknown, remainingTime)
            if (collectStatistics) {
                coreStatistics.queriesToSolverNumber++
                coreStatistics.timeInSolverQueriesMs += (markNow() - markCheckStart).inWholeMilliseconds
            }

            when (status) {
                UNKNOWN -> {
                    logger.info { "core minimization ended --- solver returned UNKNOWN" }
                    return CoreUtils.coreToSoftConstraints(unsatCore, assumptions)
                }

                SAT -> {
                    minimalUnsatCore.add(expr)
                    _minimalUnsatCoreModel.updateModel(assumptions)
                }

                else -> processUnsat(notExpr, unknown, minimalUnsatCore)
            }
        }

        logger.info { "core minimization ended --- core is minimized" }
        return CoreUtils.coreToSoftConstraints(minimalUnsatCore, assumptions)
    }

    /**
     * Get last core minimization statistics (number of queries to solver, time in solver queries (ms) etc.).
     */
    fun collectStatistics(): MinimalCoreStatistics {
        require(this::coreStatistics.isInitialized) {
            "Minimal core construction statistics is only available after core minimization launches with statistics collection enabled"
        }

        return coreStatistics
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
                if (ModelUtils.expressionIsNotTrue(ctx, model, asm.expression)) {
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
