package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KDecl
import io.ksmt.decl.KNotDecl
import io.ksmt.decl.KStringLenDecl
import io.ksmt.decl.KStringLiteralDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KStringSort

class KStringLiteralExpr internal constructor(
    ctx: KContext,
    val value: String
) : KInterpretedValue<KStringSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val decl: KStringLiteralDecl
        get() = ctx.mkStringLiteralDecl(value)

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

class KStringConcatExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KStringSort, KStringSort>(ctx) {
    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KStringSort>
        get() = ctx.mkStringConcatDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> {
        TODO("Not yet implemented")
    }

    override val sort: KStringSort = ctx.mkStringSort()

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringLenExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KIntSort, KStringSort>(ctx) {
    override val sort: KIntSort = ctx.intSort

    override val decl: KStringLenDecl
        get() = ctx.mkStringLenDecl()

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}
