package io.ksmt.solver.maxsat.solvers

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsat.constraints.SoftConstraint
import io.ksmt.solver.maxsat.utils.CoreUtils

abstract class KMaxResSolver<T>(
    private val ctx: KContext,
    private val solver: KSolver<T>,
) : KMaxSATSolver<T>(ctx, solver) where T : KSolverConfiguration {
    protected fun removeCoreAssumptions(core: List<SoftConstraint>, assumptions: MutableList<SoftConstraint>) {
        assumptions.removeAll(core)
    }

    protected fun splitCore(
        core: List<SoftConstraint>,
        assumptions: MutableList<SoftConstraint>,
    ): Pair<UInt, List<SoftConstraint>> {
        val splitCore = mutableListOf<SoftConstraint>()

        val minWeight = CoreUtils.getCoreWeight(core)
        // Add fresh soft clauses for weights that are above w.
        for (constraint in core) {
            if (constraint.weight > minWeight) {
                assumptions.add(SoftConstraint(constraint.expression, constraint.weight - minWeight))
                splitCore.add(SoftConstraint(constraint.expression, minWeight))
            } else {
                splitCore.add(SoftConstraint(constraint.expression, constraint.weight))
            }
        }

        return Pair(minWeight, splitCore)
    }
}
