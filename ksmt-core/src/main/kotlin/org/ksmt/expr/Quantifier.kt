package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
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

    override fun internHashCode(): Int = hash(body, bounds)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { body }, { bounds })
}

class KUniversalQuantifier internal constructor(
    ctx: KContext,
    body: KExpr<KBoolSort>,
    bounds: List<KDecl<*>>
) : KQuantifier(ctx, body, bounds) {
    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun printQuantifierName(): String = "forall"

    override fun internHashCode(): Int = hash(body, bounds)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { body }, { bounds })
}
