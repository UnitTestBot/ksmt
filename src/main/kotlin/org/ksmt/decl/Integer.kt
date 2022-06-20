package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkIntMod
import org.ksmt.expr.mkIntRem
import org.ksmt.expr.mkIntToReal
import org.ksmt.expr.mkIntNum
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

object KIntModDecl : KFuncDecl2<KIntSort, KIntSort, KIntSort>("intMod", KIntSort, KIntSort, KIntSort) {
    override fun apply(arg0: KExpr<KIntSort>, arg1: KExpr<KIntSort>): KExpr<KIntSort> = mkIntMod(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KIntRemDecl : KFuncDecl2<KIntSort, KIntSort, KIntSort>("intRem", KIntSort, KIntSort, KIntSort) {
    override fun apply(arg0: KExpr<KIntSort>, arg1: KExpr<KIntSort>): KExpr<KIntSort> = mkIntRem(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KIntToRealDecl : KFuncDecl1<KRealSort, KIntSort>("intToReal", KRealSort, KIntSort) {
    override fun apply(arg: KExpr<KIntSort>): KExpr<KRealSort> = mkIntToReal(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntNumDecl(val value: String) : KConstDecl<KIntSort>(value, KIntSort) {
    override fun apply(args: List<KExpr<*>>): KExpr<KIntSort> = mkIntNum(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
