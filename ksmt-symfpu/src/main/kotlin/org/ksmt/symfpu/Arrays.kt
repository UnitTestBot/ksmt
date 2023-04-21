package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.cast

internal fun KContext.mkArrayAnySort(domain: List<KSort>, range: KSort): KArraySortBase<KSort> =
    when (domain.size) {
        KArraySort.DOMAIN_SIZE -> mkArraySort(domain.single(), range)
        KArray2Sort.DOMAIN_SIZE -> {
            val (d0, d1) = domain
            mkArraySort(d0, d1, range)
        }

        KArray3Sort.DOMAIN_SIZE -> {
            val (d0, d1, d2) = domain
            mkArraySort(d0, d1, d2, range)
        }

        else -> mkArrayNSort(domain, range)
    }

internal fun KContext.mkArrayAnyLambda(
    indices: List<KDecl<*>>,
    body: KExpr<*>,
): KArrayLambdaBase<out KArraySortBase<*>, *> =
    when (indices.size) {
        KArraySort.DOMAIN_SIZE -> mkArrayLambda(indices.single(), body)
        KArray2Sort.DOMAIN_SIZE -> {
            val (i0, i1) = indices
            mkArrayLambda(i0, i1, body)
        }

        KArray3Sort.DOMAIN_SIZE -> {
            val (i0, i1, i2) = indices
            mkArrayLambda(i0, i1, i2, body)
        }

        else -> mkArrayNLambda(indices, body)
    }

internal fun <D : KSort> packToBvIfUnpacked(expr: KExpr<D>): KExpr<KSort> = with(expr.ctx) {
    ((expr as? UnpackedFp<*>)?.let { packToBv(expr) } ?: expr).cast()
}

internal fun <R : KSort> KContext.arraySelectUnpacked(sort: R, res: KExpr<R>): KExpr<R> =
    if (sort is KFpSort) {
        val resTyped: KExpr<KBvSort> = res.cast()
        unpackBiased(sort, resTyped).cast()
    } else {
        res
    }
