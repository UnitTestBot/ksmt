package io.ksmt.solver.maxsmt.utils

import io.ksmt.expr.KExpr
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.sort.KBoolSort

internal object CoreUtils {
    fun coreToSoftConstraints(core: List<KExpr<KBoolSort>>, assumptions: List<SoftConstraint>): List<SoftConstraint> {
        val softs = mutableListOf<SoftConstraint>()

        for (element in core) {
            val softElement = assumptions.find { it.expression == element }

            require(softElement != null) { "Assumptions do not contain an element from the core" }

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
