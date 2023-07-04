package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KAndDecl
import io.ksmt.decl.KDistinctDecl
import io.ksmt.decl.KEqDecl
import io.ksmt.decl.KFalseDecl
import io.ksmt.decl.KImpliesDecl
import io.ksmt.decl.KIteDecl
import io.ksmt.decl.KNotDecl
import io.ksmt.decl.KOrDecl
import io.ksmt.decl.KTrueDecl
import io.ksmt.decl.KXorDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

abstract class KAndExpr(ctx: KContext) : KApp<KBoolSort, KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KAndDecl
        get() = ctx.mkAndDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KAndBinaryExpr internal constructor(
    ctx: KContext,
    val lhs: KExpr<KBoolSort>,
    val rhs: KExpr<KBoolSort>
) : KAndExpr(ctx) {
    override val args: List<KExpr<KBoolSort>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KAndNaryExpr internal constructor(
    ctx: KContext,
    override val args: List<KExpr<KBoolSort>>
) : KAndExpr(ctx) {
    init {
        require(args.size != 2) {
            "Specialized binary AND expression should be used"
        }
    }

    override fun internHashCode(): Int = hash(args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { args }
}

abstract class KOrExpr(ctx: KContext) : KApp<KBoolSort, KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KOrDecl
        get() = ctx.mkOrDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KOrBinaryExpr internal constructor(
    ctx: KContext,
    val lhs: KExpr<KBoolSort>,
    val rhs: KExpr<KBoolSort>
) : KOrExpr(ctx) {
    override val args: List<KExpr<KBoolSort>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KOrNaryExpr internal constructor(
    ctx: KContext,
    override val args: List<KExpr<KBoolSort>>
) : KOrExpr(ctx) {
    init {
        require(args.size != 2) {
            "Specialized binary OR expression should be used"
        }
    }

    override fun internHashCode(): Int = hash(args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { args }
}

class KNotExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KBoolSort>
) : KApp<KBoolSort, KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KNotDecl
        get() = ctx.mkNotDecl()

    override val args: List<KExpr<KBoolSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KImpliesExpr internal constructor(
    ctx: KContext,
    val p: KExpr<KBoolSort>,
    val q: KExpr<KBoolSort>
) : KApp<KBoolSort, KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KImpliesDecl
        get() = ctx.mkImpliesDecl()

    override val args: List<KExpr<KBoolSort>>
        get() = listOf(p, q)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(p, q)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { p }, { q })
}

class KXorExpr internal constructor(
    ctx: KContext,
    val a: KExpr<KBoolSort>,
    val b: KExpr<KBoolSort>
) : KApp<KBoolSort, KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KXorDecl
        get() = ctx.mkXorDecl()

    override val args: List<KExpr<KBoolSort>>
        get() = listOf(a, b)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(a, b)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { a }, { b })
}

class KEqExpr<T : KSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>, val rhs: KExpr<T>
) : KApp<KBoolSort, T>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KEqDecl<T>
        get() = with(ctx) { mkEqDecl(lhs.sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KDistinctExpr<T : KSort> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<KBoolSort, T>(ctx) {
    init {
        require(args.isNotEmpty()) { "distinct requires at least a single argument" }
    }

    override val sort: KBoolSort = ctx.boolSort

    override val decl: KDistinctDecl<T>
        get() = with(ctx) { mkDistinctDecl(args.first().sort) }

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { args }
}

class KIteExpr<T : KSort> internal constructor(
    ctx: KContext,
    val condition: KExpr<KBoolSort>,
    val trueBranch: KExpr<T>,
    val falseBranch: KExpr<T>
) : KApp<T, KSort>(ctx) {

    override val decl: KIteDecl<T>
        get() = ctx.mkIteDecl(trueBranch.sort)

    override val args: List<KExpr<KSort>>
        get() = listOf(condition, trueBranch, falseBranch).uncheckedCast()

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = trueBranch.sort

    override fun internHashCode(): Int = hash(condition, trueBranch, falseBranch)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { condition }, { trueBranch }, { falseBranch })
}

class KTrue(ctx: KContext) : KInterpretedValue<KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KTrueDecl
        get() = ctx.mkTrueDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}

class KFalse(ctx: KContext) : KInterpretedValue<KBoolSort>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KFalseDecl
        get() = ctx.mkFalseDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash()
    override fun internEquals(other: Any): Boolean = structurallyEqual(other)
}
