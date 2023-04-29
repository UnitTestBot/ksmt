package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort

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
