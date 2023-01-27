package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStoreDecl<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArraySort<D, R>
) : KFuncDecl3<KArraySort<D, R>, KArraySort<D, R>, D, R>(
    ctx,
    name = "store",
    resultSort = sort,
    arg0Sort = sort,
    arg1Sort = sort.domain,
    arg2Sort = sort.range
) {
    override fun KContext.apply(
        arg0: KExpr<KArraySort<D, R>>,
        arg1: KExpr<D>, arg2: KExpr<R>
    ): KApp<KArraySort<D, R>, *> = mkArrayStore(arg0, arg1, arg2)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArraySelectDecl<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArraySort<D, R>
) : KFuncDecl2<R, KArraySort<D, R>, D>(
    ctx,
    name = "select",
    resultSort = sort.range,
    arg0Sort = sort,
    arg1Sort = sort.domain
) {
    override fun KContext.apply(
        arg0: KExpr<KArraySort<D, R>>,
        arg1: KExpr<D>
    ): KApp<R, *> = mkArraySelect(arg0, arg1)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArrayConstDecl<D : KSort, R : KSort>(
    ctx: KContext,
    sort: KArraySort<D, R>
) : KFuncDecl1<KArraySort<D, R>, R>(ctx, "const", sort, sort.range) {
    override fun KContext.apply(arg: KExpr<R>): KApp<KArraySort<D, R>, R> = mkArrayConst(sort, arg)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
