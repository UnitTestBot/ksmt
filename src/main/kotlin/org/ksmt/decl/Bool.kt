package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

object KAndDecl : KFuncDeclChain<KBoolSort, KBoolSort>("and", KBoolSort, KBoolSort) {
    override fun KContext.applyChain(args: List<KExpr<KBoolSort>>): KApp<KBoolSort, KExpr<KBoolSort>> = mkAnd(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KOrDecl : KFuncDeclChain<KBoolSort, KBoolSort>("or", KBoolSort, KBoolSort) {
    override fun KContext.applyChain(args: List<KExpr<KBoolSort>>): KApp<KBoolSort, KExpr<KBoolSort>> = mkOr(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KNotDecl : KFuncDecl1<KBoolSort, KBoolSort>("not", KBoolSort, KBoolSort) {
    override fun KContext.apply(arg: KExpr<KBoolSort>): KApp<KBoolSort, KExpr<KBoolSort>> = mkNot(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KEqDecl<T : KSort>(argSort: T) : KFuncDecl2<KBoolSort, T, T>("eq", KBoolSort, argSort, argSort) {
    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = mkEq(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIteDecl<T : KSort>(argSort: T) : KFuncDecl3<T, KBoolSort, T, T>("ite", argSort, KBoolSort, argSort, argSort) {
    override fun KContext.apply(arg0: KExpr<KBoolSort>, arg1: KExpr<T>, arg2: KExpr<T>): KApp<T, *> =
        mkIte(arg0, arg1, arg2)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KTrueDecl : KConstDecl<KBoolSort>("true", KBoolSort) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<KBoolSort, *> = mkTrue()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KFalseDecl : KConstDecl<KBoolSort>("false", KBoolSort) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<KBoolSort, *> = mkFalse()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
