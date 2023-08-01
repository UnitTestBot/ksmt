package io.ksmt.symfpu.solver

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KExpr
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.symfpu.operations.UnpackedFp
import io.ksmt.symfpu.operations.packToBv
import io.ksmt.symfpu.operations.unpack
import io.ksmt.utils.uncheckedCast

class ArraysTransform(val ctx: KContext, private val packedBvOptimizationEnabled: Boolean) {
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


    fun <R : KSort> arraySelectUnpacked(sort: R, res: KExpr<R>): KExpr<R> = with(ctx) {
        if (sort is KFpSort) {
            unpack(sort, res.uncheckedCast(), packedBvOptimizationEnabled).uncheckedCast()
        } else {
            res
        }
    }

    companion object {
        internal fun <D : KSort> packToBvIfUnpacked(expr: KExpr<D>): KExpr<D> = with(expr.ctx) {
            ((expr as? UnpackedFp<*>)?.let { packToBv(expr) } ?: expr).uncheckedCast()
        }

        private fun KContext.mkAnyArraySort(domain: List<KSort>, range: KSort): KArraySortBase<KSort> =
            mkAnyArrayOperation(
                domain,
                { d0 -> mkArraySort(d0, range) },
                { d0, d1 -> mkArraySort(d0, d1, range) },
                { d0, d1, d2 -> mkArraySort(d0, d1, d2, range) },
                { mkArrayNSort(it, range) }
            )

        fun KContext.mkAnyArrayLambda(domain: List<KDecl<*>>, body: KExpr<*>) =
            mkAnyArrayOperation(
                domain,
                { d0 -> mkArrayLambda(d0, body) },
                { d0, d1 -> mkArrayLambda(d0, d1, body) },
                { d0, d1, d2 -> mkArrayLambda(d0, d1, d2, body) },
                { mkArrayNLambda(it, body) }
            )

        fun <A : KArraySortBase<*>> KContext.mkAnyArrayStore(
            array: KExpr<A>,
            indices: List<KExpr<KSort>>,
            value: KExpr<KSort>,
        ): KExpr<out KArraySortBase<KSort>> {
            val domain = array.sort.domainSorts
            return when (domain.size) {
                KArraySort.DOMAIN_SIZE -> mkArrayStore(array.uncheckedCast(), indices.single(), value)
                KArray2Sort.DOMAIN_SIZE -> mkArrayStore(array.uncheckedCast(), indices.first(), indices.last(), value)
                KArray3Sort.DOMAIN_SIZE -> {
                    val (d0, d1, d2) = indices
                    mkArrayStore(array.uncheckedCast(), d0, d1, d2, value)
                }

                else -> mkArrayNStore(array.uncheckedCast(), indices, value)
            }
        }


        private inline fun <T, R> mkAnyArrayOperation(
            domain: List<T>,
            array1: (T) -> R,
            array2: (T, T) -> R,
            array3: (T, T, T) -> R,
            arrayN: (List<T>) -> R,
        ): R = when (domain.size) {
            KArraySort.DOMAIN_SIZE -> array1(domain.single())
            KArray2Sort.DOMAIN_SIZE -> array2(domain.first(), domain.last())
            KArray3Sort.DOMAIN_SIZE -> {
                val (d0, d1, d2) = domain
                array3(d0, d1, d2)
            }

            else -> arrayN(domain)
        }


        fun <A : KArraySortBase<*>> transformedArraySort(
            expr: KExpr<A>,
        ): A = with(expr.ctx) {
            return transformArraySort(expr.sort).uncheckedCast()
        }

        private fun KContext.transformArraySort(sort: KArraySortBase<*>): KArraySortBase<KSort> {
            val domains = sort.domainSorts.map {
                transformSortRemoveFP(it)
            }

            val prevRange = sort.range
            val range = transformSortRemoveFP(prevRange)

            return mkAnyArraySort(domains, range)
        }

        fun KContext.transformSortRemoveFP(it: KSort) = when (it) {
            is KFpSort -> {
                mkBvSort(it.exponentBits + it.significandBits)
            }

            is KArraySortBase<*> -> transformArraySort(it)

            else -> it
        }
    }
}


