package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

fun <D : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    value: KExpr<R>
) : KExpr<KArraySort<D, R>> = mkArrayStoreNoSimplify(array, index, value)

fun <D : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>
): KExpr<R> = mkArraySelectNoSimplify(array, index)
