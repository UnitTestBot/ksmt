package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkArraySelect
import org.ksmt.expr.mkArrayStore
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStoreDecl<D : KSort, R : KSort>(sort: KArraySort<D, R>) :
    KFuncDecl3<KArraySort<D, R>, KArraySort<D, R>, D, R>("store", sort, sort, sort.domain, sort.range) {
    override fun apply(arg0: KExpr<KArraySort<D, R>>, arg1: KExpr<D>, arg2: KExpr<R>): KExpr<KArraySort<D, R>> =
        mkArrayStore(arg0, arg1, arg2)
}

class KArraySelectDecl<D : KSort, R : KSort>(sort: KArraySort<D, R>) :
    KFuncDecl2<R, KArraySort<D, R>, D>("select", sort.range, sort, sort.domain) {
    override fun apply(arg0: KExpr<KArraySort<D, R>>, arg1: KExpr<D>): KExpr<R> = mkArraySelect(arg0, arg1)
}
