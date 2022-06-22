package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KArithAddDecl<T : KArithSort<T>>(argumentSort: T) : KFuncDeclChain<T, T>("arithAdd", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, KExpr<T>> = mkArithAdd(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithMulDecl<T : KArithSort<T>>(argumentSort: T) : KFuncDeclChain<T, T>("arithMul", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, KExpr<T>> = mkArithMul(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithSubDecl<T : KArithSort<T>>(argumentSort: T) : KFuncDeclChain<T, T>("arithSub", argumentSort, argumentSort) {
    override fun KContext.applyChain(args: List<KExpr<T>>): KApp<T, KExpr<T>> = mkArithSub(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithUnaryMinusDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl1<T, T>("arithUnaryMinus", argumentSort, argumentSort) {
    override fun KContext.apply(arg: KExpr<T>): KApp<T, KExpr<T>> = mkArithUnaryMinus(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithDivDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<T, T, T>("arithDiv", argumentSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkArithDiv(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithPowerDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<T, T, T>("arithPower", argumentSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkArithPower(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithLtDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithLt", KBoolSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithLt(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithLeDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithLe", KBoolSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithLe(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithGtDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithGt", KBoolSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithGt(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArithGeDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithGe", KBoolSort, argumentSort, argumentSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkArithGe(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
