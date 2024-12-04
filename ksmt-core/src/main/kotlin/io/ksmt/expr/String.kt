package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.*
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.*
import io.ksmt.utils.cast

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

class KStringToRegexExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KRegexSort, KStringSort>(ctx) {
    override val sort: KRegexSort = ctx.regexSort

    override val decl: KStringToRegexDecl
        get() = ctx.mkStringToRegexDecl()

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringInRegexExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KBoolSort, KSort>(ctx) {
    override val sort: KBoolSort = ctx.mkBoolSort()

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast())

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkStringInRegexDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KSuffixOfExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort = ctx.mkBoolSort()

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkSuffixOfDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KPrefixOfExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort = ctx.mkBoolSort()

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkPrefixOfDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}
