package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KRegexLiteralDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KRegexSort

class KRegexLiteralExpr internal constructor(
    ctx: KContext,
    val value: String
) : KInterpretedValue<KRegexSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val decl: KRegexLiteralDecl
        get() = ctx.mkRegexLiteralDecl(value)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}
