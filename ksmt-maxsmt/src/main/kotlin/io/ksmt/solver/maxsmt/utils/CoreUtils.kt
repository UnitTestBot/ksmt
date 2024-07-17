package io.ksmt.solver.maxsmt.utils

import io.ksmt.expr.KExpr
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.sort.KBoolSort

internal object CoreUtils {
    fun coreToSoftConstraints(core: List<KExpr<KBoolSort>>, assumptions: List<SoftConstraint>): List<SoftConstraint> {
        val softs = mutableListOf<SoftConstraint>()

        val potentialSofts = assumptions.filter { core.contains(it.expression) }.toMutableList()

        for (element in core) {
            val softIndex = potentialSofts.indexOfFirst { it.expression == element }
            val softElement = potentialSofts[softIndex]

            potentialSofts.removeAt(softIndex)
            softs.add(softElement)
        }

        return softs
    }

    fun getCoreWeight(core: List<SoftConstraint>): UInt {
        if (core.isEmpty()) {
            return 0u
        }

        return core.minOf { it.weight }
    }
}
