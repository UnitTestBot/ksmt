package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KArithAddDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDeclChain<T, T>(ctx, "arithAdd", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, KExpr<T>> = mkArithAdd(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithMulDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDeclChain<T, T>(ctx, "arithMul", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, KExpr<T>> = mkArithMul(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithSubDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDeclChain<T, T>(ctx, "arithSub", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, KExpr<T>> = mkArithSub(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithUnaryMinusDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl1<T, T>(ctx, "arithUnaryMinus", argumentSort, argumentSort) {
    override fun KContext.apply(arg: KExpr<T>): KApp<T, KExpr<T>> = mkArithUnaryMinus(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithDivDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<T, T, T>(ctx, "arithDiv", argumentSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkArithDiv(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithPowerDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<T, T, T>(ctx, "arithPower", argumentSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkArithPower(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithLtDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithLt", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithLt(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithLeDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithLe", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithLe(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithGtDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithGt", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithGt(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithGeDecl<T : KArithSort<T>> internal constructor(ctx: KContext, argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "arithGe", ctx.mkBoolSort(), argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithGe(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
