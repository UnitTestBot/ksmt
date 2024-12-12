package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KStringConcatDecl
import io.ksmt.decl.KStringLenDecl
import io.ksmt.decl.KStringSuffixOfDecl
import io.ksmt.decl.KStringPrefixOfDecl
import io.ksmt.decl.KStringLtDecl
import io.ksmt.decl.KStringLeDecl
import io.ksmt.decl.KStringGtDecl
import io.ksmt.decl.KStringGeDecl
import io.ksmt.decl.KStringToRegexDecl
import io.ksmt.decl.KStringInRegexDecl
import io.ksmt.decl.KStringContainsDecl
import io.ksmt.decl.KStringSingletonSubDecl
import io.ksmt.decl.KStringSubDecl
import io.ksmt.decl.KStringIndexOfDecl
import io.ksmt.decl.KStringReplaceDecl
import io.ksmt.decl.KStringReplaceAllDecl
import io.ksmt.decl.KStringReplaceWithRegexDecl
import io.ksmt.decl.KStringReplaceAllWithRegexDecl
import io.ksmt.decl.KStringIsDigitDecl
import io.ksmt.decl.KStringToCodeDecl
import io.ksmt.decl.KStringFromCodeDecl
import io.ksmt.decl.KStringToIntDecl
import io.ksmt.decl.KStringFromIntDecl
import io.ksmt.decl.KStringLiteralDecl
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KSort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRegexSort
import io.ksmt.sort.KStringSort
import io.ksmt.utils.cast

class KStringConcatExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KStringSort, KStringSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.mkStringSort()

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringConcatDecl
        get() = ctx.mkStringConcatDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringLenExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KIntSort, KStringSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override val decl: KStringLenDecl
        get() = ctx.mkStringLenDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringToRegexExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KRegexSort, KStringSort>(ctx) {
    override val sort: KRegexSort
        get() = ctx.regexSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override val decl: KStringToRegexDecl
        get() = ctx.mkStringToRegexDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KRegexSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringInRegexExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KRegexSort>
) : KApp<KBoolSort, KSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast())

    override val decl: KStringInRegexDecl
        get() = ctx.mkStringInRegexDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringSuffixOfExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringSuffixOfDecl
        get() = ctx.mkStringSuffixOfDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringPrefixOfExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringPrefixOfDecl
        get() = ctx.mkStringPrefixOfDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringLtExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringLtDecl
        get() = ctx.mkStringLtDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringLeExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringLeDecl
        get() = ctx.mkStringLeDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringGtExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringGtDecl
        get() = ctx.mkStringGtDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringGeExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringGeDecl
        get() = ctx.mkStringGeDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringContainsExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1)

    override val decl: KStringContainsDecl
        get() = ctx.mkStringContainsDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringSingletonSubExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KIntSort>
) : KApp<KStringSort, KSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast())

    override val decl: KStringSingletonSubDecl
        get() = ctx.mkStringSingletonSubDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KStringSubExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KIntSort>,
    val arg2: KExpr<KIntSort>
) : KApp<KStringSort, KSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast(), arg2.cast())

    override val decl: KStringSubDecl
        get() = ctx.mkStringSubDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, arg2)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { arg2 })
}

class KStringIndexOfExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>,
    val arg2: KExpr<KIntSort>
) : KApp<KIntSort, KSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast(), arg2.cast())

    override val decl: KStringIndexOfDecl
        get() = ctx.mkStringIndexOfDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, arg2)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { arg2 })
}

class KStringReplaceExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>,
    val arg2: KExpr<KStringSort>
) : KApp<KStringSort, KStringSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1, arg2)

    override val decl: KStringReplaceDecl
        get() = ctx.mkStringReplaceDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, arg2)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { arg2 })
}

class KStringReplaceAllExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KStringSort>,
    val arg2: KExpr<KStringSort>
) : KApp<KStringSort, KStringSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg0, arg1, arg2)

    override val decl: KStringReplaceAllDecl
        get() = ctx.mkStringReplaceAllDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, arg2)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { arg2 })
}

class KStringReplaceWithRegexExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KRegexSort>,
    val arg2: KExpr<KStringSort>
) : KApp<KStringSort, KSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast(), arg2.cast())

    override val decl: KStringReplaceWithRegexDecl
        get() = ctx.mkStringReplaceWithRegexDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, arg2)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { arg2 })
}

class KStringReplaceAllWithRegexExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KStringSort>,
    val arg1: KExpr<KRegexSort>,
    val arg2: KExpr<KStringSort>
) : KApp<KStringSort, KSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KSort>>
        get() = listOf(arg0.cast(), arg1.cast(), arg2.cast())

    override val decl: KStringReplaceAllWithRegexDecl
        get() = ctx.mkStringReplaceAllWithRegexDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, arg2)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { arg2 })
}

/*
    Maps to and from integers.
 */

class KStringIsDigitExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KBoolSort, KStringSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override val decl: KStringIsDigitDecl
        get() = ctx.mkStringIsDigitDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringToCodeExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KIntSort, KStringSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override val decl: KStringToCodeDecl
        get() = ctx.mkStringToCodeDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringFromCodeExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KIntSort>
) : KApp<KStringSort, KIntSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KIntSort>>
        get() = listOf(arg)

    override val decl: KStringFromCodeDecl
        get() = ctx.mkStringFromCodeDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringToIntExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KStringSort>
) : KApp<KIntSort, KStringSort>(ctx) {
    override val sort: KIntSort
        get() = ctx.intSort

    override val args: List<KExpr<KStringSort>>
        get() = listOf(arg)

    override val decl: KStringToIntDecl
        get() = ctx.mkStringToIntDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringFromIntExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KIntSort>
) : KApp<KStringSort, KIntSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val args: List<KExpr<KIntSort>>
        get() = listOf(arg)

    override val decl: KStringFromIntDecl
        get() = ctx.mkStringFromIntDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KStringLiteralExpr internal constructor(
    ctx: KContext,
    val value: String
) : KInterpretedValue<KStringSort>(ctx) {
    override val sort: KStringSort
        get() = ctx.stringSort

    override val decl: KStringLiteralDecl
        get() = ctx.mkStringLiteralDecl(value)

    override fun accept(transformer: KTransformerBase): KExpr<KStringSort> =
        transformer.transform(this)

    override fun print(printer: ExpressionPrinter) = with(printer) { append("\"$value\"") }

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}
