package io.ksmt.solver.maxsmt.solvers

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.utils.CoreUtils

abstract class KMaxResSolverBase<T>(
    ctx: KContext,
    solver: KSolver<out T>,
) : KMaxSMTSolverBase<T>(ctx, solver) where T : KSolverConfiguration {
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
