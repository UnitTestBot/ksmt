package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

class KIntModDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KIntSort, KIntSort, KIntSort>(
    ctx,
    "intMod",
    ctx.mkIntSort(),
    ctx.mkIntSort(),
    ctx.mkIntSort()
) {
    override fun KContext.apply(
        arg0: KExpr<KIntSort>,
        arg1: KExpr<KIntSort>
    ): KApp<KIntSort, *> = mkIntModNoSimplify(arg0, arg1)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntRemDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KIntSort, KIntSort, KIntSort>(
    ctx,
    "intRem",
    ctx.mkIntSort(),
    ctx.mkIntSort(),
    ctx.mkIntSort()
) {
    override fun KContext.apply(
        arg0: KExpr<KIntSort>,
        arg1: KExpr<KIntSort>
    ): KApp<KIntSort, *> = mkIntRemNoSimplify(arg0, arg1)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntToRealDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KRealSort, KIntSort>(
    ctx,
    "intToReal",
    ctx.mkRealSort(),
    ctx.mkIntSort()
) {
    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KRealSort, KIntSort> = mkIntToRealNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntNumDecl internal constructor(
    ctx: KContext,
    val value: String
) : KConstDecl<KIntSort>(ctx, value, ctx.mkIntSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KIntSort, *> = ctx.mkIntNum(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
