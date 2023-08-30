package io.ksmt.solver.maxsat.utils

import io.ksmt.expr.KExpr
import io.ksmt.solver.maxsat.constraints.SoftConstraint
import io.ksmt.sort.KBoolSort

internal object CoreUtils {
    fun coreToSoftConstraints(core: List<KExpr<KBoolSort>>, assumptions: List<SoftConstraint>): List<SoftConstraint> {
        val softs = mutableListOf<SoftConstraint>()
        for (soft in assumptions) {
            if (core.any { it.internEquals(soft.expression) }) {
                softs.add(soft)
            }
        }

        require(core.size == softs.size) {
            "Unsat core size [${core.size}] was not equal to corresponding soft constraints size [${softs.size}]"
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
