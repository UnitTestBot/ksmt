package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.KApp
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

object KIntModDecl : KFuncDecl2<KIntSort, KIntSort, KIntSort>("intMod", KIntSort, KIntSort, KIntSort) {
    override fun KContext.apply(arg0: KExpr<KIntSort>, arg1: KExpr<KIntSort>): KApp<KIntSort, *> = mkIntMod(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KIntRemDecl : KFuncDecl2<KIntSort, KIntSort, KIntSort>("intRem", KIntSort, KIntSort, KIntSort) {
    override fun KContext.apply(arg0: KExpr<KIntSort>, arg1: KExpr<KIntSort>): KApp<KIntSort, *> = mkIntRem(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

object KIntToRealDecl : KFuncDecl1<KRealSort, KIntSort>("intToReal", KRealSort, KIntSort) {
    override fun KContext.apply(arg: KExpr<KIntSort>): KApp<KRealSort, KExpr<KIntSort>> = mkIntToReal(arg)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KIntNumDecl(val value: String) : KConstDecl<KIntSort>(value, KIntSort) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<KIntSort, *> = mkIntNum(value)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
