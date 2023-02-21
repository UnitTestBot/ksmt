package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KArithAddDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDeclChain<T, T>(ctx, "arithAdd", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, T> = mkArithAddNoSimplify(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithMulDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDeclChain<T, T>(ctx, "arithMul", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, T> = mkArithMulNoSimplify(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithSubDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDeclChain<T, T>(ctx, "arithSub", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, T> = mkArithSubNoSimplify(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithUnaryMinusDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl1<T, T>(ctx, "arithUnaryMinus", argumentSort, argumentSort) {
    override fun KContext.apply(arg: KExpr<T>): KApp<T, T> = mkArithUnaryMinusNoSimplify(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithDivDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<T, T, T>(ctx, "arithDiv", argumentSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkArithDivNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithPowerDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<T, T, T>(ctx, "arithPower", argumentSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkArithPowerNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithLtDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithLt", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithLtNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithLeDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithLe", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithLeNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithGtDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithGt", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithGtNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithGeDecl<T : KArithSort> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithGe", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithGeNoSimplify(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
