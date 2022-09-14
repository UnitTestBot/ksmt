package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

class KRealToIntDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KIntSort, KRealSort>(
    ctx,
    "realToInt",
    ctx.mkIntSort(),
    ctx.mkRealSort()
) {
    override fun KContext.apply(arg: KExpr<KRealSort>): KApp<KIntSort, KExpr<KRealSort>> = mkRealToInt(arg)

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
    override fun KContext.apply(arg: KExpr<KRealSort>): KApp<KBoolSort, KExpr<KRealSort>> = mkRealIsInt(arg)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRealNumDecl internal constructor(
    ctx: KContext,
    val value: String
) : KConstDecl<KRealSort>(ctx, value, ctx.mkRealSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KRealSort, *> = ctx.mkRealNum(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
