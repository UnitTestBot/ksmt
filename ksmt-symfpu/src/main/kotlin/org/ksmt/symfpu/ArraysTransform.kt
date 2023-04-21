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

class ArraysTransform(val ctx: KContext) {

    val mapFpToBvDeclImpl = mutableMapOf<KDecl<KFpSort>, KDecl<KBvSort>>()

    fun mkArrayAnySort(domain: List<KSort>, range: KSort): KArraySortBase<KSort> = with(ctx) {
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
    }

    fun mkArrayAnyLambda(
        indices: List<KDecl<*>>,
        body: KExpr<*>,
    ): KArrayLambdaBase<out KArraySortBase<*>, *> = with(ctx) {
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
    }

    fun transformDeclList(
        decls: List<KDecl<*>>,
    ): List<KDecl<*>> = with(ctx) {
        decls.map {
            if (it.sort is KFpSort) {
                val asFp: KDecl<KFpSort> = it.cast()
                mapFpToBvDeclImpl.getOrPut(asFp) {
                    mkConst(asFp.name + "!tobv!", mkBvSort(
                        asFp.sort.exponentBits + asFp.sort.significandBits)).decl
                }
            } else it
        }
    }

    fun <R : KSort> arraySelectUnpacked(sort: R, res: KExpr<R>): KExpr<R> = with(ctx) {
        if (sort is KFpSort) {
            val resTyped: KExpr<KBvSort> = res.cast()
            unpackBiased(sort, resTyped).cast()
        } else {
            res
        }
    }

    companion object {
        internal fun <D : KSort> packToBvIfUnpacked(expr: KExpr<D>): KExpr<KSort> = with(expr.ctx) {
            ((expr as? UnpackedFp<*>)?.let { packToBv(expr) } ?: expr).cast()
        }
    }
}


