package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KApp
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort

class KAndDecl internal constructor(
    ctx: KContext
) : KFuncDeclChain<KBoolSort, KBoolSort>(ctx, "and", ctx.mkBoolSort(), ctx.mkBoolSort()) {
    override fun KContext.applyChain(args: List<KExpr<KBoolSort>>): KApp<KBoolSort, KBoolSort> = mkAndNoSimplify(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KOrDecl internal constructor(
    ctx: KContext
) : KFuncDeclChain<KBoolSort, KBoolSort>(ctx, "or", ctx.mkBoolSort(), ctx.mkBoolSort()) {
    override fun KContext.applyChain(args: List<KExpr<KBoolSort>>): KApp<KBoolSort, KBoolSort> = mkOrNoSimplify(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KNotDecl internal constructor(
    ctx: KContext
) : KFuncDecl1<KBoolSort, KBoolSort>(ctx, "not", ctx.mkBoolSort(), ctx.mkBoolSort()) {
    override fun KContext.apply(arg: KExpr<KBoolSort>): KApp<KBoolSort, KBoolSort> = mkNotNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KImpliesDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KBoolSort, KBoolSort>(
    ctx,
    name = "implies",
    ctx.mkBoolSort(),
    ctx.mkBoolSort(),
    ctx.mkBoolSort()
) {
    override fun KContext.apply(
        arg0: KExpr<KBoolSort>,
        arg1: KExpr<KBoolSort>
    ): KApp<KBoolSort, KBoolSort> = mkImpliesNoSimplify(arg0, arg1)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KXorDecl internal constructor(
    ctx: KContext
) : KFuncDecl2<KBoolSort, KBoolSort, KBoolSort>(
    ctx,
    name = "xor",
    ctx.mkBoolSort(),
    ctx.mkBoolSort(),
    ctx.mkBoolSort()
) {
    override fun KContext.apply(
        arg0: KExpr<KBoolSort>,
        arg1: KExpr<KBoolSort>
    ): KApp<KBoolSort, KBoolSort> = mkXorNoSimplify(arg0, arg1)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KEqDecl<T : KSort> internal constructor(
    ctx: KContext,
    argSort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "=", ctx.mkBoolSort(), argSort, argSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, T> = mkEqNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KDistinctDecl<T : KSort> internal constructor(
    ctx: KContext,
    argSort: T
) : KFuncDeclChain<KBoolSort, T>(ctx, "distinct", ctx.mkBoolSort(), argSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<KBoolSort, T> = mkDistinctNoSimplify(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIteDecl<T : KSort> internal constructor(
    ctx: KContext,
    argSort: T
) : KFuncDecl3<T, KBoolSort, T, T>(ctx, "ite", argSort, ctx.mkBoolSort(), argSort, argSort) {
    override fun KContext.apply(
        arg0: KExpr<KBoolSort>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = mkIteNoSimplify(arg0, arg1, arg2)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KTrueDecl internal constructor(
    ctx: KContext
) : KConstDecl<KBoolSort>(ctx, "true", ctx.mkBoolSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBoolSort, *> = ctx.mkTrue()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFalseDecl internal constructor(
    ctx: KContext
) : KConstDecl<KBoolSort>(ctx, "false", ctx.mkBoolSort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBoolSort, *> = ctx.mkFalse()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
