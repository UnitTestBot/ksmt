package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KArithAddDecl
import org.ksmt.decl.KArithDivDecl
import org.ksmt.decl.KArithGeDecl
import org.ksmt.decl.KArithGtDecl
import org.ksmt.decl.KArithLeDecl
import org.ksmt.decl.KArithLtDecl
import org.ksmt.decl.KArithMulDecl
import org.ksmt.decl.KArithPowerDecl
import org.ksmt.decl.KArithSubDecl
import org.ksmt.decl.KArithUnaryMinusDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KAddArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "add requires at least a single argument" }
    }

    override fun decl(): KArithAddDecl<T> = with(ctx) { mkArithAddDecl(sort) }

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = args.first().sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += args.first()
    }
}

class KMulArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "mul requires at least a single argument" }
    }

    override fun decl(): KArithMulDecl<T> = with(ctx) { mkArithMulDecl(sort) }

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = args.first().sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += args.first()
    }
}

class KSubArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "sub requires at least a single argument" }
    }

    override fun decl(): KArithSubDecl<T> = with(ctx) { mkArithSubDecl(sort) }

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = args.first().sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += args.first()
    }
}

class KUnaryMinusArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val arg: KExpr<T>
) : KApp<T, KExpr<T>>(ctx) {

    override fun decl(): KArithUnaryMinusDecl<T> = with(ctx) { mkArithUnaryMinusDecl(sort) }

    override val args: List<KExpr<T>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = arg.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg
    }
}

class KDivArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, KExpr<T>>(ctx) {

    override fun decl(): KArithDivDecl<T> = with(ctx) { mkArithDivDecl(sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = lhs.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += lhs
    }
}

class KPowerArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, KExpr<T>>(ctx) {

    override fun decl(): KArithPowerDecl<T> = with(ctx) { mkArithPowerDecl(sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = lhs.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += lhs
    }
}

class KLtArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KArithLtDecl<T> = with(ctx) { mkArithLtDecl(lhs.sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KLeArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KArithLeDecl<T> = with(ctx) { mkArithLeDecl(lhs.sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KGtArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KArithGtDecl<T> = with(ctx) { mkArithGtDecl(lhs.sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KGeArithExpr<T : KArithSort<T>> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KArithGeDecl<T> = with(ctx) { mkArithGeDecl(lhs.sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}
