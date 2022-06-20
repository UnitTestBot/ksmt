package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkAnd
import org.ksmt.expr.mkNot
import org.ksmt.expr.mkOr
import org.ksmt.expr.mkEq
import org.ksmt.expr.mkTrue
import org.ksmt.expr.mkFalse
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

object KAndDecl : KFuncDeclChain<KBoolSort, KBoolSort>("and", KBoolSort, KBoolSort) {
    override fun applyChain(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = mkAnd(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KOrDecl : KFuncDeclChain<KBoolSort, KBoolSort>("or", KBoolSort, KBoolSort) {
    override fun applyChain(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = mkOr(args)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KNotDecl : KFuncDecl1<KBoolSort, KBoolSort>("not", KBoolSort, KBoolSort) {
    override fun apply(arg: KExpr<KBoolSort>): KExpr<KBoolSort> = mkNot(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KEqDecl<T : KSort>(argSort: T) : KFuncDecl2<KBoolSort, T, T>("eq", KBoolSort, argSort, argSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> = mkEq(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KTrueDecl : KConstDecl<KBoolSort>("true", KBoolSort) {
    override fun apply(args: List<KExpr<*>>): KExpr<KBoolSort> = mkTrue()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KFalseDecl : KConstDecl<KBoolSort>("false", KBoolSort) {
    override fun apply(args: List<KExpr<*>>): KExpr<KBoolSort> = mkFalse()
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
