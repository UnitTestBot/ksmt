package io.ksmt.solver.maxsmt.solvers

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.KSolverStatus.UNKNOWN
import io.ksmt.solver.KSolverStatus.UNSAT
import io.ksmt.solver.maxsmt.KMaxSMTContext
import io.ksmt.solver.maxsmt.KMaxSMTContext.Strategy.PrimalMaxRes
import io.ksmt.solver.maxsmt.KMaxSMTResult
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.solver.maxsmt.utils.CoreUtils
import io.ksmt.solver.maxsmt.utils.TimerUtils
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.mkConst
import kotlin.time.Duration
import kotlin.time.TimeSource.Monotonic.markNow

class KPMResSolver<T : KSolverConfiguration>(private val ctx: KContext, private val solver: KSolver<out T>) :
    KMaxResSolver<T>(ctx, solver) {
    private var collectStatistics = false
    private val maxSmtCtx = KMaxSMTContext(
        strategy = PrimalMaxRes,
        preferLargeWeightConstraintsForCores = false,
        minimizeCores = false,
        getMultipleCores = false,
    )

    override fun checkMaxSMT(timeout: Duration, collectStatistics: Boolean): KMaxSMTResult {
        val markCheckMaxSMTStart = markNow()

        if (TimerUtils.timeoutExceeded(timeout)) {
            error("Timeout must be positive but was [${timeout.inWholeSeconds} s]")
        }

        this.collectStatistics = collectStatistics

        if (this.collectStatistics) {
            maxSMTStatistics = KMaxSMTStatistics(maxSmtCtx)
            maxSMTStatistics.timeoutMs = timeout.inWholeMilliseconds
        }

        val maxSMTResult = runMaxSMTLogic(timeout)

        if (this.collectStatistics) {
            maxSMTStatistics.elapsedTimeMs = markCheckMaxSMTStart.elapsedNow().inWholeMilliseconds
        }

        return maxSMTResult
    }

    override fun checkSubOptMaxSMT(timeout: Duration, collectStatistics: Boolean): KMaxSMTResult {
        TODO("Not yet implemented")
    }

    private fun runMaxSMTLogic(timeout: Duration): KMaxSMTResult {
        val markHardConstraintsCheckStart = markNow()
        val hardConstraintsStatus = solver.check(timeout)

        if (collectStatistics) {
            maxSMTStatistics.queriesToSolverNumber++
            maxSMTStatistics.timeInSolverQueriesMs += markHardConstraintsCheckStart.elapsedNow().inWholeMilliseconds
        }

        if (softConstraints.isEmpty()) {
            return KMaxSMTResult(
                listOf(),
                hardConstraintsStatus,
                hardConstraintsStatus != UNKNOWN,
            )
        }

        if (hardConstraintsStatus == UNSAT) {
            return KMaxSMTResult(listOf(), hardConstraintsStatus, true)
        } else if (hardConstraintsStatus == UNKNOWN) {
            return KMaxSMTResult(listOf(), hardConstraintsStatus, false)
        }

        solver.push()

        var i = 0
        var formula = softConstraints.toMutableList()

        while (true) {
            val checkRemainingTime = TimerUtils.computeRemainingTime(timeout, markHardConstraintsCheckStart)

            if (TimerUtils.timeoutExceeded(checkRemainingTime)) {
                solver.pop()
                return KMaxSMTResult(listOf(), hardConstraintsStatus, false)
            }

            val markCheckSatStart = markNow()
            val (solverStatus, unsatCore, model) =
                checkSat(formula, checkRemainingTime)

            if (collectStatistics) {
                maxSMTStatistics.queriesToSolverNumber++
                maxSMTStatistics.timeInSolverQueriesMs += markCheckSatStart.elapsedNow().inWholeMilliseconds
            }

            if (solverStatus == SAT) {
                solver.pop()
                return processSat(model!!)
            }
            else if (solverStatus == UNKNOWN) {
                solver.pop()
                return KMaxSMTResult(emptyList(), SAT, false)
            }

            val (weight, splitUnsatCore) = splitUnsatCore(formula, unsatCore)

            val (formulaReified, reificationLiterals) =
                reifyUnsatCore(formula, splitUnsatCore, i, weight)

            unionReificationLiterals(reificationLiterals)

            formula = applyMaxRes(formulaReified, reificationLiterals, i, weight)

            i++
        }
    }

    /**
     * Split all soft constraints from the unsat core into two groups:
     * - constraints with the weight equal to the minimum of the unsat core soft constraint weights
     * - constraints with the weight equal to old weight - minimum weight
     *
     * @return a pair of minimum weight and a list of unsat core soft constraints with minimum weight.
     */
    private fun splitUnsatCore(
        formula: MutableList<SoftConstraint>,
        unsatCore: List<KExpr<KBoolSort>>,
    ): Pair<UInt, List<SoftConstraint>> {
        val unsatCoreSoftConstraints = CoreUtils.coreToSoftConstraints(unsatCore, formula)
        removeCoreAssumptions(unsatCoreSoftConstraints, formula)
        return splitCore(unsatCoreSoftConstraints, formula)
    }

    /**
     * Reify unsat core soft constraints with literals.
     */
    private fun reifyUnsatCore(
        formula: MutableList<SoftConstraint>,
        unsatCore: List<SoftConstraint>,
        iter: Int,
        weight: UInt,
    ): Pair<MutableList<SoftConstraint>, List<KExpr<KBoolSort>>> = with(ctx) {
        val literalsToReify = mutableListOf<KExpr<KBoolSort>>()

        unsatCore.forEachIndexed { index, coreElement ->
            if (coreElement.weight == weight) {
                formula.remove(coreElement)

                val coreElementExpr = coreElement.expression
                val literalToReify = coreElementExpr.sort.mkConst("*$iter:$index")
                val constraintToReify = coreElementExpr eq !literalToReify

                assert(constraintToReify)

                literalsToReify.add(literalToReify)
            }
        }

        return Pair(formula, literalsToReify)
    }

    private fun unionReificationLiterals(reificationLiterals: List<KExpr<KBoolSort>>) = with(ctx) {
        when (reificationLiterals.size) {
            1 -> assert(reificationLiterals.first())
            2 -> assert(reificationLiterals[0] or reificationLiterals[1])
            else -> assert(reificationLiterals.reduce { x, y -> x or y })
        }
    }

    /**
     * Apply MaxRes rule.
     */
    private fun applyMaxRes(
        formula: MutableList<SoftConstraint>,
        literalsToReify: List<KExpr<KBoolSort>>,
        iter: Int,
        weight: UInt,
    ): MutableList<SoftConstraint> =
        with(ctx) {
            literalsToReify.forEachIndexed { index, literal ->
                val indexLast = literalsToReify.lastIndex

                if (index < indexLast) {
                    val sort = literal.sort
                    val currentLiteralToReifyDisjunction = sort.mkConst("#$iter:$index")
                    val nextLiteralToReifyDisjunction = sort.mkConst("#$iter:${index + 1}")

                    val disjunction =
                        when (indexLast - index) {
                            // The second element is omitted as it is an empty disjunction.
                            1 -> literalsToReify[index + 1]
                            else -> literalsToReify[index + 1] or nextLiteralToReifyDisjunction
                        }

                    assert(currentLiteralToReifyDisjunction eq disjunction)

                    formula.add(SoftConstraint(!currentLiteralToReifyDisjunction or !literal, weight))
                }
            }

            return formula
        }

    private fun processSat(model: KModel): KMaxSMTResult {
        val satSoftConstraints = getSatSoftConstraintsByModel(model)
        return KMaxSMTResult(satSoftConstraints, SAT, true)
    }

    /**
     * Check on satisfiability hard constraints with assumed soft constraints.
     *
     * @return a triple of solver status, unsat core (if exists, empty list otherwise) and model
     * (if exists, null otherwise).
     */
    private fun checkSat(assumptions: List<SoftConstraint>, timeout: Duration):
        Triple<KSolverStatus, List<KExpr<KBoolSort>>, KModel?> =
        when (val status = solver.checkWithAssumptions(assumptions.map { x -> x.expression }, timeout)) {
            SAT -> Triple(status, listOf(), solver.model())
            UNSAT -> Triple(status, solver.unsatCore(), null)
            UNKNOWN -> Triple(status, listOf(), null)
        }
}
