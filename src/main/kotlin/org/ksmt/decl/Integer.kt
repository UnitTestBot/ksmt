package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

class KIntModDecl(ctx: KContext) :
    KFuncDecl2<KIntSort, KIntSort, KIntSort>(ctx, "intMod", ctx.mkIntSort(), ctx.mkIntSort(), ctx.mkIntSort()) {
    override fun KContext.apply(arg0: KExpr<KIntSort>, arg1: KExpr<KIntSort>): KApp<KIntSort, *> = mkIntMod(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntRemDecl(ctx: KContext) :
    KFuncDecl2<KIntSort, KIntSort, KIntSort>(ctx, "intRem", ctx.mkIntSort(), ctx.mkIntSort(), ctx.mkIntSort()) {
    override fun KContext.apply(arg0: KExpr<KIntSort>, arg1: KExpr<KIntSort>): KApp<KIntSort, *> = mkIntRem(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntToRealDecl(ctx: KContext) :
    KFuncDecl1<KRealSort, KIntSort>(ctx, "intToReal", ctx.mkRealSort(), ctx.mkIntSort()) {
    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KRealSort, KExpr<KIntSort>> = mkIntToReal(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntNumDecl(ctx: KContext, val value: String) : KConstDecl<KIntSort>(ctx, value, ctx.mkIntSort()) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<KIntSort, *> = mkIntNum(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
