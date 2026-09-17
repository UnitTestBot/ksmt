package io.ksmt.expr.rewrite

import io.ksmt.KContext
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.sort.KBoolSort

class KQuantifierSubstitutor(ctx: KContext) : KExprSubstitutor(ctx) {

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> =
        transformExpr(expr)

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> =
        transformExpr(expr)
}
