package io.ksmt.solver.maxsmt

import io.ksmt.KContext
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrBinaryExpr
import io.ksmt.expr.KOrNaryExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.mkConst
import kotlin.time.Duration

// TODO: solver type must be KSolver<KSolverConfiguration> but the code does not work with it
class KMaxSMTSolver(private val ctx: KContext, private val solver: KZ3Solver) : KSolver<KSolverConfiguration> {
    private val softConstraints = mutableListOf<SoftConstraint>()

    // Enum checking max SAT state (last check status, was not checked, invalid (soft assertions changed))
    // Should I support push/pop for soft constraints?

    fun assertSoft(expr: KExpr<KBoolSort>, weight: Int) {
        softConstraints.add(SoftConstraint(expr, weight))
    }

    // TODO: return soft constraints
    // TODO: add timeout?
    fun checkMaxSMT(): Pair<KModel?, Int> {
        require(softConstraints.isNotEmpty()) { "Soft constraints list should not be empty" }

        // Should I check every time on satisfiability?
        // У них в солверах есть last checked status.
        // Should I add timeout?
        val status = solver.check()

        if (status == KSolverStatus.UNSAT) {
            error("Conjunction of asserted formulas is UNSAT")
        } else if (status == KSolverStatus.UNKNOWN) {
            // TODO: handle this case
        }

        var i = 0
        var formula = softConstraints.toMutableList()

        while (true) {
            val (solverStatus, unsatCore, model) = solveSMT(formula)

            if (solverStatus == KSolverStatus.SAT) {
                return Pair(model, i)
            } else if (solverStatus == KSolverStatus.UNKNOWN) {
                // TODO: implement
            }

            val (formulaReified, reificationVariables) =
                reifyCore(formula, getUnsatCoreOfConstraints(formula, unsatCore), i)

            // TODO, FIX: Для одного странно использовать KOrNaryExpr
            this.assert(KOrNaryExpr(ctx, reificationVariables))

            formula = applyMaxRes(formulaReified, reificationVariables)

            ++i
        }
    }

    // Returns issat, unsat core (?) and assignment
    private fun solveSMT(assumptions: List<SoftConstraint>): Triple<KSolverStatus, List<KExpr<KBoolSort>>, KModel?> {
        val status = solver.checkWithAssumptions(assumptions.map { x -> x.constraint })

        if (status == KSolverStatus.SAT) {
            return Triple(status, listOf(), solver.model())
        } else if (status == KSolverStatus.UNSAT) {
            return Triple(status, solver.unsatCore(), null)
        }

        return Triple(status, listOf(), null)
    }

    private fun reifyCore(formula: MutableList<SoftConstraint>, unsatCore: List<SoftConstraint>, i: Int)
            : Pair<MutableList<SoftConstraint>, List<KExpr<KBoolSort>>> {
        val literalsToReify = mutableListOf<KExpr<KBoolSort>>()

        for (coreElement in unsatCore.withIndex()) {
            if (coreElement.value.weight == 1) {
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
                this.assert(constraintToReify)

                formula.add(SoftConstraint(KNotExpr(ctx, literalToReify), 1))

                literalsToReify.add(literalToReify)
            }
        }

        return Pair(formula, literalsToReify)
    }

    private fun applyMaxRes(formula: MutableList<SoftConstraint>, literalsToReify: List<KExpr<KBoolSort>>)
            : MutableList<SoftConstraint> {
        for (indexedLiteral in literalsToReify.withIndex()) {
            // TODO: here we should use restrictions from the article for MaxRes

            formula.removeIf { x ->
                x.constraint.internEquals(KNotExpr(ctx, indexedLiteral.value)) &&
                    x.weight == 1
            }

            val index = indexedLiteral.index
            val indexLast = literalsToReify.size - 1

            if (index < indexLast) {
                val disjunction = KOrNaryExpr(
                    ctx,
                    literalsToReify.subList(index + 1, indexLast),
                )

                val literalToReifyDisjunction = ctx.boolSort.mkConst("d$indexedLiteral")

                this.assert(
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
                        1,
                    ),
                )
            } else {
                // Здесь добавляем пустой дизъюнкт, но по факту это не нужно делать (т.к. потом его удалим)
            }
        }

        return formula
    }

    private fun getUnsatCoreOfConstraints(formula: MutableList<SoftConstraint>, unsatCore: List<KExpr<KBoolSort>>)
            : List<SoftConstraint> {
        val unsatCoreOfConstraints = mutableListOf<SoftConstraint>()

        for (coreElement in unsatCore) {
            val softConstraint = formula.find { x -> x.constraint == coreElement }
            softConstraint?.let { unsatCoreOfConstraints.add(it) }
        }

        return unsatCoreOfConstraints
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
