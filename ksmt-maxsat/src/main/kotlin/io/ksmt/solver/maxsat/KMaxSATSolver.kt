package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KTrue
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.mkConst
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.time.Duration

class KMaxSATSolver<T>(private val ctx: KContext, private val solver: KSolver<T>) : KSolver<KSolverConfiguration>
    where T : KSolverConfiguration {
    private val scopeManager = MaxSATScopeManager()
    private var softConstraints = mutableListOf<SoftConstraint>()

    /**
     * Assert softly an expression with weight (aka soft constraint) into solver.
     *
     * @see checkMaxSAT
     * */
    fun assertSoft(expr: KExpr<KBoolSort>, weight: Int) {
        require(weight > 0) { "Soft constraint weight cannot be equal to $weight as it must be greater than 0" }

        val softConstraint = SoftConstraint(expr, weight)
        softConstraints.add(softConstraint)
        scopeManager.incrementSoft()
    }

    /**
     * Solve maximum satisfiability problem.
     *
     * @throws NotImplementedError
     */
    suspend fun checkMaxSAT(timeout: Duration = Duration.INFINITE): KMaxSATResult {
        var hardConstraintsSatStatus = KSolverStatus.UNKNOWN
        var currentMaxSATResult: Triple<KSolverStatus?, List<KExpr<KBoolSort>>, KModel?> =
            Triple(null, listOf(), null)

        solver.push()

        val maxSATResult = withTimeoutOrNull(timeout) {
            if (softConstraints.isEmpty()) {
                return@withTimeoutOrNull KMaxSATResult(
                    listOf(),
                    solver.check(),
                    maxSATSucceeded = true,
                    timeoutExceeded = false,
                )
            }

            val status = solver.check()
            hardConstraintsSatStatus = status

            if (status == KSolverStatus.UNSAT) {
                return@withTimeoutOrNull KMaxSATResult(
                    listOf(),
                    status,
                    maxSATSucceeded = true,
                    timeoutExceeded = false,
                )
            } else if (status == KSolverStatus.UNKNOWN) {
                return@withTimeoutOrNull KMaxSATResult(
                    listOf(),
                    status,
                    maxSATSucceeded = false,
                    timeoutExceeded = false,
                )
            }

            var i = 0
            var formula = softConstraints.toMutableList()

            unionSoftConstraintsWithSameExpressions(formula)

            while (true) {
                val (solverStatus, unsatCore, model) = checkSAT(formula)

                if (solverStatus == KSolverStatus.UNKNOWN) {
                    throw NotImplementedError()
                }

                currentMaxSATResult = Triple(solverStatus, unsatCore, model)

                if (solverStatus == KSolverStatus.SAT) {
                    return@withTimeoutOrNull handleSat(model!!)
                }

                val (weight, splitUnsatCore) = splitUnsatCore(formula, unsatCore)

                val (formulaReified, reificationLiterals) =
                    reifyUnsatCore(formula, splitUnsatCore, i, weight)

                unionReificationLiterals(reificationLiterals)

                formula = applyMaxRes(formulaReified, reificationLiterals, i, weight)

                i++
            }
        }

        solver.pop()

        if (maxSATResult == null) {
            val (solverStatus, _, model) = currentMaxSATResult

            return when (solverStatus) {
                null -> KMaxSATResult(
                    listOf(),
                    hardConstraintsSatStatus,
                    maxSATSucceeded = false,
                    timeoutExceeded = true,
                )
                KSolverStatus.SAT -> handleSat(model!!)
                KSolverStatus.UNSAT -> throw NotImplementedError()
                KSolverStatus.UNKNOWN -> throw NotImplementedError()
            }
        }

        return maxSATResult as KMaxSATResult
    }

    /**
     * Split all soft constraints from the unsat core into two groups:
     * - constraints with the weight equal to the minimum of the unsat core soft constraint weights
     * - constraints with the weight equal to old weight - minimum weight
     *
     * @return a pair of minimum weight and a list of unsat core soft constraints with minimum weight.
     */
    private fun splitUnsatCore(formula: MutableList<SoftConstraint>, unsatCore: List<KExpr<KBoolSort>>)
    : Pair<Int, List<SoftConstraint>> {
        // Filters soft constraints from the unsat core.
        val unsatCoreSoftConstraints =
            formula.filter { x -> unsatCore.any { x.expression.internEquals(it) } }

        val minWeight = unsatCoreSoftConstraints.minBy { it.weight }.weight

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
     * Check on satisfiability hard constraints with assumed soft constraints.
     *
     * @return a triple of solver status, unsat core (if exists, empty list otherwise) and model
     * (if exists, null otherwise).
     */
    private fun checkSAT(assumptions: List<SoftConstraint>): Triple<KSolverStatus, List<KExpr<KBoolSort>>, KModel?> =
        when (val status = solver.checkWithAssumptions(assumptions.map { x -> x.expression })) {
            KSolverStatus.SAT -> Triple(status, listOf(), solver.model())
            KSolverStatus.UNSAT -> Triple(status, solver.unsatCore(), null)
            KSolverStatus.UNKNOWN -> Triple(status, listOf(), null)
        }

    /**
     * Reify unsat core soft constraints with literals.
     */
    private fun reifyUnsatCore(
        formula: MutableList<SoftConstraint>,
        unsatCore: List<SoftConstraint>,
        iter: Int,
        weight: Int,
    ): Pair<MutableList<SoftConstraint>, List<KExpr<KBoolSort>>> = with(ctx) {
        val literalsToReify = mutableListOf<KExpr<KBoolSort>>()

        unsatCore.forEachIndexed { index, coreElement ->
            if (coreElement.weight == weight) {
                formula.remove(coreElement)

                val coreElementExpr = coreElement.expression
                val literalToReify = coreElementExpr.sort.mkConst("*$iter$index")

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
        weight: Int,
    )
    : MutableList<SoftConstraint> = with(ctx) {
        literalsToReify.forEachIndexed { index, literal ->
            val indexLast = literalsToReify.lastIndex

            if (index < indexLast) {
                val sort = literal.sort
                val currentLiteralToReifyDisjunction = sort.mkConst("#$iter$index")
                val nextLiteralToReifyDisjunction = sort.mkConst("#$iter${index + 1}")

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
        return KMaxSATResult(satSoftConstraints, KSolverStatus.SAT, maxSATSucceeded = true, timeoutExceeded = false)
    }

    private fun getSatSoftConstraintsByModel(model: KModel): List<SoftConstraint> {
        return softConstraints.filter { model.eval(it.expression).internEquals(KTrue(ctx)) }
    }

    override fun configure(configurator: KSolverConfiguration.() -> Unit) {
        solver.configure(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) {
        solver.assert(expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        solver.assertAndTrack(expr)
    }

    override fun push() {
        solver.push()
        scopeManager.push()
    }

    override fun pop(n: UInt) {
        solver.pop(n)
        softConstraints = scopeManager.pop(n, softConstraints)
    }

    override fun check(timeout: Duration): KSolverStatus {
        return solver.check(timeout)
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        return solver.checkWithAssumptions(assumptions, timeout)
    }

    override fun model(): KModel {
        return solver.model()
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        return solver.unsatCore()
    }

    override fun reasonOfUnknown(): String {
        return solver.reasonOfUnknown()
    }

    override fun interrupt() {
        solver.interrupt()
    }

    override fun close() {
        solver.close()
    }
}
