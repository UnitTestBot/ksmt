package io.ksmt.solver.maxsat

import io.ksmt.KContext
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrNaryExpr
import io.ksmt.expr.KTrue
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.mkConst
import kotlin.time.Duration

class KMaxSATSolver<T>(private val ctx: KContext, private val solver: KSolver<T>) : KSolver<KSolverConfiguration>
        where T : KSolverConfiguration {
    private val softConstraints = mutableListOf<SoftConstraint>()

    fun assertSoft(expr: KExpr<KBoolSort>, weight: Int) {
        require(weight > 0) { "Soft constraint weight must be greater than 0" }
        softConstraints.add(SoftConstraint(expr, weight))
    }

    // TODO: add timeout support
    fun checkMaxSAT(): MaxSATResult {
        if (softConstraints.isEmpty()) {
            return MaxSATResult(listOf(), solver.check(), true)
        }

        val status = solver.check()

        if (status == KSolverStatus.UNSAT) {
            return MaxSATResult(listOf(), status, true)
        } else if (status == KSolverStatus.UNKNOWN) {
            return MaxSATResult(listOf(), status, false)
        }

        var i = 0
        var formula = softConstraints.toMutableList()

        unionSoftConstraintsWithSameExpressions(formula)

        while (true) {
            val (solverStatus, unsatCore, model) = solveSAT(formula)

            if (solverStatus == KSolverStatus.SAT) {
                // TODO: can I simplify this expression?
                val satSoftConstraints =
                        softConstraints.filter { model?.eval(it.constraint)?.internEquals(KTrue(ctx)) == true }
                return MaxSATResult(satSoftConstraints, solverStatus, true)
            } else if (solverStatus == KSolverStatus.UNKNOWN) {
                // TODO: implement
            }

            val (weight, splitUnsatCore) = splitUnsatCoreSoftConstraints(formula, unsatCore)

            val (formulaReified, reificationVariables) =
                    reifyCore(formula, splitUnsatCore, i, weight)

            when (reificationVariables.size) {
                1 -> assert(reificationVariables.first())
                2 -> assert(KOrBinaryExpr(ctx, reificationVariables[0], reificationVariables[1]))
                else -> assert(KOrNaryExpr(ctx, reificationVariables))
            }

            formula = applyMaxRes(formulaReified, reificationVariables, weight)

            ++i
        }
    }

    /**
     * Splits all soft constraints from the unsat core into two groups:
     * - constraints with the weight equal to the minimum from the unsat core soft constraint weight
     * - constraints with the weight equal to old weight - minimum weight
     *
     * Returns a pair of minimum weight and a list of unsat core soft constraints with minimum weight.
     */
    private fun splitUnsatCoreSoftConstraints(formula: MutableList<SoftConstraint>, unsatCore: List<KExpr<KBoolSort>>)
            : Pair<Int, List<SoftConstraint>> {
        // Filters soft constraints from the unsat core.
        val unsatCoreSoftConstraints  =
                formula.filter { x -> unsatCore.any { x.constraint.internEquals(it) } }.toMutableList()

        val minWeight = unsatCoreSoftConstraints.minBy { it.weight }.weight

        // Splits every soft constraint from the unsat core with the weight greater than
        // minimum weight into two soft constraints with the same expression: with minimum weight and
        // with the weight equal to old weight - minimum weight.
        unsatCoreSoftConstraints.forEach { x ->
            if (x.weight > minWeight) {
                formula.add(SoftConstraint(x.constraint, minWeight))
                formula.add(SoftConstraint(x.constraint, x.weight - minWeight))
                formula.removeIf { it.weight == x.weight && it.constraint.internEquals(x.constraint) }
            }
        }

        return Pair(minWeight, formula.filter { x -> x.weight == minWeight && unsatCore.any { x.constraint.internEquals(it) } })
    }

    /**
     * Unions soft constraints with same expressions into a single soft constraint. The soft constraint weight will be
     * equal to the sum of old soft constraints weights.
     */
    private fun unionSoftConstraintsWithSameExpressions(formula: MutableList<SoftConstraint>) {
        var i = 0

        while (i < formula.size) {
            val currentConstraint = formula[i].constraint

            val sameConstraints = formula.filter { it.constraint.internEquals(currentConstraint) }

            // Unions soft constraints with same expressions into a single soft constraint.
            if (sameConstraints.size > 1) {
                val sameConstraintsWeightSum = sameConstraints.sumOf { it.weight }

                formula.removeAll(sameConstraints)
                formula.add(SoftConstraint(currentConstraint, sameConstraintsWeightSum))
            }

            ++i
        }
    }

    private fun solveSAT(assumptions: List<SoftConstraint>): Triple<KSolverStatus, List<KExpr<KBoolSort>>, KModel?> =
        when (val status = solver.checkWithAssumptions(assumptions.map { x -> x.constraint })) {
            KSolverStatus.SAT -> Triple(status, listOf(), solver.model())
            KSolverStatus.UNSAT -> Triple(status, solver.unsatCore(), null)
            KSolverStatus.UNKNOWN -> Triple(status, listOf(), null)
        }

    private fun reifyCore(formula: MutableList<SoftConstraint>, unsatCore: List<SoftConstraint>, i: Int, weight: Int)
            : Pair<MutableList<SoftConstraint>, List<KExpr<KBoolSort>>> {
        val literalsToReify = mutableListOf<KExpr<KBoolSort>>()

        for (coreElement in unsatCore.withIndex()) {
            if (coreElement.value.weight == weight) {
                formula.remove(coreElement.value)

                val coreElementConstraint = coreElement.value.constraint
                val literalToReify =
                    ctx.boolSort.mkConst("b$i${coreElement.index}")

                val constraintToReify = KEqExpr(
                    ctx,
                    coreElementConstraint,
                    KNotExpr(ctx, literalToReify),
                )

                assert(constraintToReify)

                formula.add(SoftConstraint(KNotExpr(ctx, literalToReify), weight))

                literalsToReify.add(literalToReify)
            }
        }

        return Pair(formula, literalsToReify)
    }

    private fun applyMaxRes(formula: MutableList<SoftConstraint>, literalsToReify: List<KExpr<KBoolSort>>, weight: Int)
            : MutableList<SoftConstraint> {
        for (indexedLiteral in literalsToReify.withIndex()) {
            // TODO: here we should use restrictions from the article for MaxRes

            formula.removeIf { it.constraint.internEquals(KNotExpr(ctx, indexedLiteral.value)) &&
                    it.weight == weight
            }

            val index = indexedLiteral.index
            val indexLast = literalsToReify.lastIndex

            if (index < indexLast) {
                val disjunction =
                    // We do not take the current element (from the next to the last)
                    when (indexLast - index) {
                        1 -> literalsToReify[index + 1]
                        2 -> KOrBinaryExpr(ctx, literalsToReify[index + 1], literalsToReify[index + 2])
                        else -> KOrNaryExpr(
                            ctx,
                            literalsToReify.subList(index + 1, indexLast + 1),
                        )
                    }

                // TODO, FIX: не будет ли коллизии при последующих запусках?
                val literalToReifyDisjunction = ctx.boolSort.mkConst("d$index")

                assert(
                    KEqExpr(
                        ctx,
                        literalToReifyDisjunction,
                        disjunction,
                    ),
                )

                formula.add(
                    SoftConstraint(
                        KOrBinaryExpr(
                            ctx,
                            KNotExpr(ctx, indexedLiteral.value),
                            KNotExpr(ctx, literalToReifyDisjunction),
                        ),
                        weight,
                    ),
                )
            }
        }

        return formula
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
    }

    override fun pop(n: UInt) {
        solver.pop(n)
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
