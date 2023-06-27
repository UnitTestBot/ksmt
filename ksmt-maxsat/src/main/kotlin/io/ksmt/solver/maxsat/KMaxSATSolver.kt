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
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.mkConst
import kotlin.time.Duration

// TODO: solver type must be KSolver<KSolverConfiguration> but the code does not work with it
class KMaxSATSolver(private val ctx: KContext, private val solver: KZ3Solver) : KSolver<KSolverConfiguration> {
    private val softConstraints = mutableListOf<SoftConstraint>()

    fun assertSoft(expr: KExpr<KBoolSort>, weight: Int) {
        require(weight > 0) { "Soft constraint weight must be greater than 0" }
        softConstraints.add(SoftConstraint(expr, weight))
    }

    // TODO: add timeout?
    fun checkMaxSMT(): Pair<KSolverStatus, List<SoftConstraint>> {
        if (softConstraints.isEmpty()) {
            return Pair(solver.check(), listOf())
        }

        val status = solver.check()

        if (status == KSolverStatus.UNSAT) {
            error("Conjunction of asserted formulas is UNSAT")
        } else if (status == KSolverStatus.UNKNOWN) {
            // TODO: implement
        }

        var i = 0
        var formula = softConstraints.toMutableList()

        while (true) {
            val (solverStatus, unsatCore, model) = solveSAT(formula)

            if (solverStatus == KSolverStatus.SAT) {
                // TODO: can I simplify this expression?
                val satSoftConstraints =
                        softConstraints.filter { model?.eval(it.constraint)?.internEquals(KTrue(ctx)) == true }
                return Pair(solverStatus, satSoftConstraints)
            } else if (solverStatus == KSolverStatus.UNKNOWN) {
                // TODO: implement
            }

            val (weight, splitUnsatCore) = splitCoreSoftConstraints(formula, unsatCore)

            val (formulaReified, reificationVariables) =
                reifyCore(formula, splitUnsatCore, i, weight)

            // TODO: а если их 0, может ли быть такое?
            when (reificationVariables.size) {
                1 -> assert(reificationVariables.first())
                2 -> assert(KOrBinaryExpr(ctx, reificationVariables[0], reificationVariables[1]))
                else -> assert(KOrNaryExpr(ctx, reificationVariables))
            }

            formula = applyMaxRes(formulaReified, reificationVariables, weight)

            ++i
        }
    }

    private fun splitCoreSoftConstraints(formula: MutableList<SoftConstraint>, unsatCore: List<KExpr<KBoolSort>>)
            : Pair<Int, List<SoftConstraint>> {
        val coreSoftConstraints  = formula.filter { x -> unsatCore.find { x.constraint.internEquals(it) } != null }
        // Here we union soft constraints from core with the same expr

        val coreSoftConstraintsUnioned = coreSoftConstraints.toMutableList()

        var i = 0
        while (i < coreSoftConstraintsUnioned.size) {
            val currentConstraint = coreSoftConstraintsUnioned[i].constraint

            val sameConstraints =
                    coreSoftConstraintsUnioned.filter { it.constraint.internEquals(currentConstraint) }

            if (sameConstraints.size > 1) {
                val sameConstraintsWeightSum = sameConstraints.sumOf { it.weight }
                coreSoftConstraintsUnioned.removeAll(sameConstraints)
                coreSoftConstraintsUnioned.add(SoftConstraint(currentConstraint, sameConstraintsWeightSum))

                formula.removeAll(sameConstraints)
                formula.add(SoftConstraint(currentConstraint, sameConstraintsWeightSum))
            }

            ++i
        }

        val minWeight = coreSoftConstraintsUnioned.minBy { it.weight }.weight

        coreSoftConstraintsUnioned.forEach { x ->
            if (x.weight > minWeight) {
                formula.add(SoftConstraint(x.constraint, minWeight))
                formula.add(SoftConstraint(x.constraint, x.weight - minWeight))
                formula.removeIf { it.weight == x.weight && it.constraint == x.constraint }
            }
        }

        return Pair(minWeight, formula.filter { x -> x.weight == minWeight && unsatCore.find { x.constraint.internEquals(it) } != null })
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
                // TODO: как реализовать переобозначение? Что если формула встречается как подформула в других формулах?
                val literalToReify =
                    ctx.boolSort.mkConst("b$i${coreElement.index}")

                val constraintToReify = KEqExpr(
                    ctx,
                    coreElementConstraint,
                    KNotExpr(ctx, literalToReify),
                )
                // TODO: Переобозначить и остальные элементы в b_i_j
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
