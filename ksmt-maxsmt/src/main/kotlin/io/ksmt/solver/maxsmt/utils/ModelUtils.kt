package io.ksmt.solver.maxsmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.sort.KBoolSort

internal object ModelUtils {
    fun expressionIsFalse(ctx: KContext, model: KModel, expression: KExpr<KBoolSort>) =
        model.eval(expression, true) == ctx.falseExpr

    fun getModelCost(ctx: KContext, model: KModel, softConstraints: List<SoftConstraint>): UInt {
        var upper = 0u

        for (soft in softConstraints) {
            if (expressionIsFalse(ctx, model, soft.expression)) {
                upper += soft.weight
            }
        }

        return upper
    }

    fun getCorrectionSet(ctx: KContext, model: KModel, softConstraints: List<SoftConstraint>): List<SoftConstraint> {
        val correctionSet = mutableListOf<SoftConstraint>()

        for (constr in softConstraints) {
            if (expressionIsFalse(ctx, model, constr.expression)) {
                correctionSet.add(constr)
            }
        }

        return correctionSet
    }
}
