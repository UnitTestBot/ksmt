package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort

class KExistentialQuantifier internal constructor(
    ctx: KContext,
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(ctx, body, bounds) {
    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun printQuantifierName(): String = "exists"
}

class KUniversalQuantifier internal constructor(
    ctx: KContext,
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(ctx, body, bounds) {
    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun printQuantifierName(): String = "forall"
}
