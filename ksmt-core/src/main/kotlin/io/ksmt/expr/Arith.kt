package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KArithAddDecl
import io.ksmt.decl.KArithDivDecl
import io.ksmt.decl.KArithGeDecl
import io.ksmt.decl.KArithGtDecl
import io.ksmt.decl.KArithLeDecl
import io.ksmt.decl.KArithLtDecl
import io.ksmt.decl.KArithMulDecl
import io.ksmt.decl.KArithPowerDecl
import io.ksmt.decl.KArithSubDecl
import io.ksmt.decl.KArithUnaryMinusDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KBoolSort

class KAddArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, T>(ctx) {
    init {
        require(args.isNotEmpty()) { "add requires at least a single argument" }
    }

    override val decl: KArithAddDecl<T>
        get() = ctx.mkArithAddDecl(sort)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = args.first().sort

    override fun internHashCode(): Int = hash(args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { args }
}

class KMulArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, T>(ctx) {
    init {
        require(args.isNotEmpty()) { "mul requires at least a single argument" }
    }

    override val decl: KArithMulDecl<T>
        get() = ctx.mkArithMulDecl(sort)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = args.first().sort

    override fun internHashCode(): Int = hash(args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { args }
}

class KSubArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, T>(ctx) {
    init {
        require(args.isNotEmpty()) { "sub requires at least a single argument" }
    }

    override val decl: KArithSubDecl<T>
        get() = ctx.mkArithSubDecl(sort)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = args.first().sort

    override fun internHashCode(): Int = hash(args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { args }
}

class KUnaryMinusArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val arg: KExpr<T>
) : KApp<T, T>(ctx) {

    override val decl: KArithUnaryMinusDecl<T>
        get() = ctx.mkArithUnaryMinusDecl(sort)

    override val args: List<KExpr<T>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = arg.sort

    override fun internHashCode(): Int = hash(arg)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { arg }
}

class KDivArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, T>(ctx) {

    override val decl: KArithDivDecl<T>
        get() = ctx.mkArithDivDecl(sort)

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = lhs.sort

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KPowerArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, T>(ctx) {

    override val decl: KArithPowerDecl<T>
        get() = ctx.mkArithPowerDecl(sort)

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T = lhs.sort

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KLtArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, T>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KArithLtDecl<T>
        get() = ctx.mkArithLtDecl(lhs.sort)

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KLeArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, T>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KArithLeDecl<T>
        get() = ctx.mkArithLeDecl(lhs.sort)

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KGtArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, T>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KArithGtDecl<T>
        get() = ctx.mkArithGtDecl(lhs.sort)

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}

class KGeArithExpr<T : KArithSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, T>(ctx) {
    override val sort: KBoolSort = ctx.boolSort

    override val decl: KArithGeDecl<T>
        get() = ctx.mkArithGeDecl(lhs.sort)

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(lhs, rhs)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { lhs }, { rhs })
}
