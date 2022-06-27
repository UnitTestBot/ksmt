package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStoreDecl<D : KSort, R : KSort>(ctx: KContext, sort: KArraySort<D, R>) :
    KFuncDecl3<KArraySort<D, R>, KArraySort<D, R>, D, R>(ctx, "store", sort, sort, sort.domain, sort.range) {
    override fun KContext.apply(arg0: KExpr<KArraySort<D, R>>, arg1: KExpr<D>, arg2: KExpr<R>): KApp<KArraySort<D, R>, *> =
        mkArrayStore(arg0, arg1, arg2)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArraySelectDecl<D : KSort, R : KSort>(ctx: KContext, sort: KArraySort<D, R>) :
    KFuncDecl2<R, KArraySort<D, R>, D>(ctx, "select", sort.range, sort, sort.domain) {
    override fun KContext.apply(arg0: KExpr<KArraySort<D, R>>, arg1: KExpr<D>): KApp<R, *> = mkArraySelect(arg0, arg1)
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
