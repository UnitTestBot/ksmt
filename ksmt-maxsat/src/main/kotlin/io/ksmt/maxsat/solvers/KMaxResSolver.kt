package io.ksmt.maxsat.solvers

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.maxsat.KMaxSATResult
import io.ksmt.maxsat.SoftConstraint
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.mkConst
import kotlin.time.Duration

class KMaxResSolver<T : KSolverConfiguration>(private val ctx: KContext, private val solver: KSolver<T>)
    : KMaxSATSolver<T>(ctx, solver) {
    private var currentMaxSATResult: Triple<KSolverStatus?, List<KExpr<KBoolSort>>, KModel?> =
        Triple(null, listOf(), null)

    /**
     * Solve maximum satisfiability problem.
     *
     * @throws NotImplementedError
     */
    override fun checkMaxSAT(timeout: Duration): KMaxSATResult {
        if (timeout.isNegative() || timeout == Duration.ZERO) {
            error("Timeout must be positive but was [${timeout.inWholeSeconds} s]")
        }

        solver.push()

        val maxSATResult = runMaxSATLogic(timeout)

        solver.pop()

        currentMaxSATResult = Triple(KSolverStatus.UNKNOWN, listOf(), null)

        if (softConstraints.isEmpty()) {
            return maxSATResult
        }

        // TODO: get max SAT soft constraints subset
        if (!maxSATResult.maxSATSucceeded) {
            val (solverStatus, _, model) = currentMaxSATResult

            return when (solverStatus) {
                KSolverStatus.SAT -> handleSat(model!!)
                KSolverStatus.UNSAT -> throw NotImplementedError()
                KSolverStatus.UNKNOWN -> throw NotImplementedError()
                else -> error("Unexpected status: $solverStatus")
            }
        }

        return maxSATResult
    }

    private fun runMaxSATLogic(timeout: Duration): KMaxSATResult {
        val clockStart = System.currentTimeMillis()

        if (softConstraints.isEmpty()) {
            val hardConstraintsStatus = solver.check(timeout)

            return KMaxSATResult(
                listOf(),
                hardConstraintsStatus,
                hardConstraintsStatus != KSolverStatus.UNKNOWN,
            )
        }

        val status = solver.check(timeout)

        if (status == KSolverStatus.UNSAT) {
            return KMaxSATResult(listOf(), status, true)
        } else if (status == KSolverStatus.UNKNOWN) {
            return KMaxSATResult(listOf(), status, false)
        }

        var i = 0
        var formula = softConstraints.toMutableList()

        unionSoftConstraintsWithSameExpressions(formula)

        while (true) {
            val softConstraintsCheckRemainingTime = computeRemainingTime(timeout, clockStart)

            if (softConstraintsCheckRemainingTime.isNegative() || softConstraintsCheckRemainingTime == Duration.ZERO) {
                return KMaxSATResult(listOf(), status, false)
            }

            val (solverStatus, unsatCore, model) =
                checkSAT(formula, softConstraintsCheckRemainingTime)

            if (solverStatus == KSolverStatus.UNKNOWN) {
                // TODO: get max SAT soft constraints subset
                return KMaxSATResult(listOf(), status, false)
            }

            currentMaxSATResult = Triple(solverStatus, unsatCore, model)

            if (solverStatus == KSolverStatus.SAT) {
                return handleSat(model!!)
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
    private fun splitUnsatCore(formula: MutableList<SoftConstraint>, unsatCore: List<KExpr<KBoolSort>>)
            : Pair<UInt, List<SoftConstraint>> {
        // Filters soft constraints from the unsat core.
        val unsatCoreSoftConstraints =
            formula.filter { x -> unsatCore.any { x.expression.internEquals(it) } }

        val minWeight = unsatCoreSoftConstraints.minOf { it.weight }

        val unsatCoreSoftConstraintsSplit = mutableListOf<SoftConstraint>()

        unsatCoreSoftConstraints.forEach { x ->
            if (x.weight > minWeight) {
                val minWeightedSoftConstraint = SoftConstraint(x.expression, minWeight)
                formula.add(minWeightedSoftConstraint)
                formula.add(SoftConstraint(x.expression, x.weight - minWeight))
                formula.removeIf { it.weight == x.weight && it.expression == x.expression }

                unsatCoreSoftConstraintsSplit.add(minWeightedSoftConstraint)
            } else {
                unsatCoreSoftConstraintsSplit.add(x)
            }
        }

        return Pair(minWeight, unsatCoreSoftConstraintsSplit)
    }

    /**
     * Union soft constraints with same expressions into a single soft constraint.
     *
     * The new soft constraint weight will be equal to the sum of old soft constraints weights.
     */
    private fun unionSoftConstraintsWithSameExpressions(formula: MutableList<SoftConstraint>) {
        val exprToRepetitionsMap = mutableMapOf<KExpr<KBoolSort>, Int>()

        formula.forEach {
            if (exprToRepetitionsMap.containsKey(it.expression)) {
                exprToRepetitionsMap[it.expression] = exprToRepetitionsMap[it.expression]!! + 1
            } else {
                exprToRepetitionsMap[it.expression] = 1
            }
        }

        exprToRepetitionsMap.forEach { (expr, repetitions) ->
            if (repetitions > 1) {
                val repeatedExpressions = formula.filter { it.expression == expr }

                formula.removeAll(repeatedExpressions)
                val repeatedExpressionsWeightsSum = repeatedExpressions.sumOf { it.weight }
                formula.add(SoftConstraint(expr, repeatedExpressionsWeightsSum))
            }
        }
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
    )
            : MutableList<SoftConstraint> = with(ctx) {
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

    private fun handleSat(model: KModel): KMaxSATResult {
        val satSoftConstraints = getSatSoftConstraintsByModel(model)
        return KMaxSATResult(satSoftConstraints, KSolverStatus.SAT, maxSATSucceeded = true)
    }
}