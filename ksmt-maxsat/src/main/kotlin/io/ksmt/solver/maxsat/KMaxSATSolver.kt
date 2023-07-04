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
        require(weight > 0) { "Soft constraint weight must be greater than 0" }

        val softConstraint = SoftConstraint(expr, weight)
        softConstraints.add(softConstraint)
        scopeManager.incrementSoft()
    }

    /**
     * Solve maximum satisfiability problem.
     */
    suspend fun checkMaxSAT(timeout: Duration = Duration.INFINITE): KMaxSATResult {
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
                    return@withTimeoutOrNull handleUnknown(timeoutExceeded = false)
                }

                currentMaxSATResult = Triple(solverStatus, unsatCore, model)

                if (solverStatus == KSolverStatus.SAT) {
                    return@withTimeoutOrNull handleSat(model!!)
                }

                val (weight, splitUnsatCore) = splitUnsatCore(formula, unsatCore)

                val (formulaReified, reificationVariables) =
                    reifyUnsatCore(formula, splitUnsatCore, i, weight)

                unionReificationVariables(reificationVariables)

                formula = applyMaxRes(formulaReified, reificationVariables, i, weight)

                i++
            }
        }

        solver.pop()

        if (maxSATResult == null) {
            val (solverStatus, unsatCore, model) = currentMaxSATResult

            return when (solverStatus) {
                null -> KMaxSATResult(listOf(), KSolverStatus.UNKNOWN, maxSATSucceeded = false, timeoutExceeded = true)
                KSolverStatus.SAT -> handleSat(model!!)
                KSolverStatus.UNSAT -> handleUnsat(unsatCore)
                KSolverStatus.UNKNOWN -> handleUnknown(timeoutExceeded = true)
            }
        }

        return maxSATResult as KMaxSATResult
    }

    /**
     * Split all soft constraints from the unsat core into two groups:
     * - constraints with the weight equal to the minimum of the unsat core soft constraint weights
     * - constraints with the weight equal to old weight - minimum weight
     *
     * Returns a pair of minimum weight and a list of unsat core soft constraints with minimum weight.
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
                val minWeightSoftConstraint = SoftConstraint(x.expression, minWeight)
                formula.add(minWeightSoftConstraint)
                formula.add(SoftConstraint(x.expression, x.weight - minWeight))
                formula.removeIf { it.weight == x.weight && it.expression == x.expression }

                unsatCoreSoftConstraintsSplit.add(minWeightSoftConstraint)
            } else {
                unsatCoreSoftConstraintsSplit.add(x)
            }
        }

        return Pair(minWeight, unsatCoreSoftConstraintsSplit)
    }

    /**
     * Union soft constraints with same expressions into a single soft constraint. The new soft constraint weight
     * will be equal to the sum of old soft constraints weights.
     */
    private fun unionSoftConstraintsWithSameExpressions(formula: MutableList<SoftConstraint>) {
        var i = 0

        while (i < formula.size) {
            val currentExpr = formula[i].expression

            val similarConstraints = formula.filter { it.expression.internEquals(currentExpr) }

            // Unions soft constraints with same expressions into a single soft constraint.
            if (similarConstraints.size > 1) {
                val similarConstraintsWeightsSum = similarConstraints.sumOf { it.weight }

                formula.removeAll(similarConstraints)
                formula.add(SoftConstraint(currentExpr, similarConstraintsWeightsSum))
            }

            i++
        }
    }

    /**
     * Check on satisfiability hard constraints with assumed soft constraints.
     *
     * Returns a triple of solver status, unsat core (if exists, empty list otherwise) and model
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

        for (coreElement in unsatCore.withIndex()) {
            if (coreElement.value.weight == weight) {
                formula.remove(coreElement.value)

                val coreElementExpr = coreElement.value.expression
                val literalToReify = coreElementExpr.sort.mkConst("*$iter${coreElement.index}")

                val constraintToReify = coreElementExpr eq !literalToReify

                assert(constraintToReify)

                literalsToReify.add(literalToReify)
            }
        }

        return Pair(formula, literalsToReify)
    }

    private fun unionReificationVariables(reificationVariables: List<KExpr<KBoolSort>>) = with(ctx) {
        when (reificationVariables.size) {
            1 -> assert(reificationVariables.first())
            2 -> assert(reificationVariables[0] or reificationVariables[1])
            else -> assert(reificationVariables.reduce { x, y -> x or y })
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
        for (indexedLiteral in literalsToReify.withIndex()) {
            val index = indexedLiteral.index
            val indexLast = literalsToReify.lastIndex

            if (index < indexLast) {
                val sort = indexedLiteral.value.sort
                val currentLiteralToReifyDisjunction = sort.mkConst("#$iter$index")
                val nextLiteralToReifyDisjunction = sort.mkConst("#$iter${index + 1}")

                val disjunction =
                    when (indexLast - index) {
                        // The second element is omitted as it is an empty disjunction.
                        1 -> literalsToReify[index + 1]
                        else -> literalsToReify[index + 1] or nextLiteralToReifyDisjunction
                    }

                assert(currentLiteralToReifyDisjunction eq disjunction)

                formula.add(SoftConstraint(!currentLiteralToReifyDisjunction or !indexedLiteral.value, weight))
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

    private fun handleUnsat(unsatCore: List<KExpr<KBoolSort>>): KMaxSATResult {
        val satSoftConstraints = softConstraints.filter { x -> unsatCore.any { x.expression.internEquals(it) } }
        return KMaxSATResult(satSoftConstraints, KSolverStatus.SAT, maxSATSucceeded = false, timeoutExceeded = true)
    }

    private fun handleUnknown(timeoutExceeded: Boolean): KMaxSATResult {
        return KMaxSATResult(listOf(), KSolverStatus.SAT, maxSATSucceeded = false, timeoutExceeded)
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
