package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.mkRealIsInt
import org.ksmt.expr.mkRealNum
import org.ksmt.expr.mkRealToInt
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

object KRealToIntDecl : KFuncDecl1<KIntSort, KRealSort>("realToInt", KIntSort, KRealSort) {
    override fun apply(arg: KExpr<KRealSort>): KApp<KIntSort, KExpr<KRealSort>> = mkRealToInt(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KRealIsIntDecl : KFuncDecl1<KBoolSort, KRealSort>("realIsInt", KBoolSort, KRealSort) {
    override fun apply(arg: KExpr<KRealSort>): KApp<KBoolSort, KExpr<KRealSort>> = mkRealIsInt(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KRealNumDecl(val value: String) : KConstDecl<KRealSort>(value, KRealSort) {
    override fun apply(args: List<KExpr<*>>): KApp<KRealSort, *> = mkRealNum(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
