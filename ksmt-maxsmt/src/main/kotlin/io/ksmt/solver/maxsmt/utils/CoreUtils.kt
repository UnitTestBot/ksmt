package io.ksmt.solver.maxsmt.utils

import io.ksmt.expr.KExpr
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.sort.KBoolSort

internal object CoreUtils {
    fun coreToSoftConstraints(core: List<KExpr<KBoolSort>>, assumptions: List<SoftConstraint>): List<SoftConstraint> {
        val uniqueCoreElements = mutableListOf<KExpr<KBoolSort>>()
        core.forEach {
            if (!uniqueCoreElements.any { u -> u.internEquals(it) }) {
                uniqueCoreElements.add(it)
            }
        }

        val softs = mutableListOf<SoftConstraint>()
        for (soft in assumptions) {
            if (uniqueCoreElements.any { it.internEquals(soft.expression) }) {
                softs.add(soft)
            }
        }

        require(uniqueCoreElements.size == softs.size) {
            "Unsat core size [${uniqueCoreElements.size}] was not equal to corresponding " +
                "soft constraints size [${softs.size}]"
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
