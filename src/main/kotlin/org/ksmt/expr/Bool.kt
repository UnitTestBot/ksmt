package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.*
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


class KAndExpr internal constructor(
    ctx: KContext,
    override val args: List<KExpr<KBoolSort>>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KAndDecl = ctx.mkAndDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KOrExpr internal constructor(
    ctx: KContext,
    override val args: List<KExpr<KBoolSort>>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KOrDecl = ctx.mkOrDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KNotExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KBoolSort>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KNotDecl = ctx.mkNotDecl()
    override val args: List<KExpr<KBoolSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KEqExpr<T : KSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>, val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KEqDecl<T> = with(ctx) { mkEqDecl(lhs.sort) }
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KIteExpr<T : KSort> internal constructor(
    ctx: KContext,
    val condition: KExpr<KBoolSort>,
    val trueBranch: KExpr<T>,
    val falseBranch: KExpr<T>
) : KApp<T, KExpr<*>>(ctx) {
    override fun sort(): T = with(ctx) { trueBranch.sort }
    override fun decl(): KIteDecl<T> = with(ctx) { mkIteDecl(trueBranch.sort) }
    override val args: List<KExpr<*>>
        get() = listOf(condition, trueBranch, falseBranch)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KTrue(ctx: KContext) : KApp<KBoolSort, KExpr<*>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KTrueDecl = ctx.mkTrueDecl()
    override val args = emptyList<KExpr<*>>()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KFalse(ctx: KContext) : KApp<KBoolSort, KExpr<*>>(ctx) {
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun decl(): KFalseDecl = ctx.mkFalseDecl()
    override val args = emptyList<KExpr<*>>()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
