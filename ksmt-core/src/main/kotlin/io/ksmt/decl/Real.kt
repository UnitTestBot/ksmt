package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KApp
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort

class KRealToIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KRealSort>(
    ctx,
    "realToInt",
    ctx.mkIntSort(),
    ctx.mkRealSort()
) {
    override fun KContext.apply(arg: KExpr<KRealSort>): KApp<KIntSort, KRealSort> = mkRealToIntNoSimplify(arg)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRealIsIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KBoolSort, KRealSort>(
    ctx,
    "realIsInt",
    ctx.mkBoolSort(),
    ctx.mkRealSort()
) {
    override fun KContext.apply(arg: KExpr<KRealSort>): KApp<KBoolSort, KRealSort> = mkRealIsIntNoSimplify(arg)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRealNumDecl internal constructor(
    ctx: KContext,
    val value: String
) : KConstDecl<KRealSort>(ctx, value, ctx.mkRealSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KRealSort, *> = ctx.mkRealNum(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
