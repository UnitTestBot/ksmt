package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

class KAndDecl(ctx: KContext) : KFuncDeclChain<KBoolSort, KBoolSort>(ctx, "and", ctx.mkBoolSort(), ctx.mkBoolSort()) {
    override fun KContext.applyChain(args: List<KExpr<KBoolSort>>): KApp<KBoolSort, KExpr<KBoolSort>> = mkAnd(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KOrDecl(ctx: KContext) : KFuncDeclChain<KBoolSort, KBoolSort>(ctx, "or", ctx.mkBoolSort(), ctx.mkBoolSort()) {
    override fun KContext.applyChain(args: List<KExpr<KBoolSort>>): KApp<KBoolSort, KExpr<KBoolSort>> = mkOr(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KNotDecl(ctx: KContext) : KFuncDecl1<KBoolSort, KBoolSort>(ctx, "not", ctx.mkBoolSort(), ctx.mkBoolSort()) {
    override fun KContext.apply(arg: KExpr<KBoolSort>): KApp<KBoolSort, KExpr<KBoolSort>> = mkNot(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KEqDecl<T : KSort>(ctx: KContext, argSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "eq", ctx.mkBoolSort(), argSort, argSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkEq(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIteDecl<T : KSort>(ctx: KContext, argSort: T) :
    KFuncDecl3<T, KBoolSort, T, T>(ctx, "ite", argSort, ctx.mkBoolSort(), argSort, argSort) {
    override fun KContext.apply(arg0: KExpr<KBoolSort>, arg1: KExpr<T>, arg2: KExpr<T>): KApp<T, *> =
        mkIte(arg0, arg1, arg2)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KTrueDecl(ctx: KContext) : KConstDecl<KBoolSort>(ctx, "true", ctx.mkBoolSort()) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<KBoolSort, *> = mkTrue()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFalseDecl(ctx: KContext) : KConstDecl<KBoolSort>(ctx, "false", ctx.mkBoolSort()) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<KBoolSort, *> = mkFalse()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
