package io.ksmt.solver.maxsmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.sort.KBoolSort

internal object ModelUtils {
    fun expressionIsNotTrue(ctx: KContext, model: KModel, expression: KExpr<KBoolSort>) =
        model.eval(expression, true) != ctx.trueExpr

    fun getModelCost(ctx: KContext, model: KModel, softConstraints: List<SoftConstraint>): ULong {
        var upper = 0uL

        for (soft in softConstraints) {
            if (expressionIsNotTrue(ctx, model, soft.expression)) {
                upper += soft.weight
            }
        }

        return upper
    }

    fun getCorrectionSet(ctx: KContext, model: KModel, softConstraints: List<SoftConstraint>): List<SoftConstraint> {
        val correctionSet = mutableListOf<SoftConstraint>()

        for (constr in softConstraints) {
            if (expressionIsNotTrue(ctx, model, constr.expression)) {
                correctionSet.add(constr)
            }
        }

        return correctionSet
    }
}
