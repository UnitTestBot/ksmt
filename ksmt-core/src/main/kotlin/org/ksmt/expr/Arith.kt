package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.*
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KAddArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "add requires at least a single argument" }
    }

    override fun sort(): T = with(ctx) { args.first().sort }
    override fun decl(): KArithAddDecl<T> = with(ctx) { mkArithAddDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KMulArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "mul requires at least a single argument" }
    }

    override fun sort(): T = with(ctx) { args.first().sort }
    override fun decl(): KArithMulDecl<T> = with(ctx) { mkArithMulDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KSubArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "sub requires at least a single argument" }
    }

    override fun sort(): T = with(ctx) { args.first().sort }
    override fun decl(): KArithSubDecl<T> = with(ctx) { mkArithSubDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KUnaryMinusArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val arg: KExpr<T>
) : KApp<T, KExpr<T>>(ctx) {
    override fun sort(): T = with(ctx) { arg.sort }
    override fun decl(): KArithUnaryMinusDecl<T> = with(ctx) { mkArithUnaryMinusDecl(sort) }
    override val args: List<KExpr<T>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KDivArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, KExpr<T>>(ctx) {
    override fun sort(): T = with(ctx) { lhs.sort }
    override fun decl(): KArithDivDecl<T> = with(ctx) { mkArithDivDecl(sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KPowerArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, KExpr<T>>(ctx) {
    override fun sort(): T = with(ctx) { lhs.sort }
    override fun decl(): KArithPowerDecl<T> = with(ctx) { mkArithPowerDecl(sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KLtArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KArithLtDecl<T> = with(ctx) { mkArithLtDecl(lhs.sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KLeArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KArithLeDecl<T> = with(ctx) { mkArithLeDecl(lhs.sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KGtArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KArithGtDecl<T> = with(ctx) { mkArithGtDecl(lhs.sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KGeArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KArithGeDecl<T> = with(ctx) { mkArithGeDecl(lhs.sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
