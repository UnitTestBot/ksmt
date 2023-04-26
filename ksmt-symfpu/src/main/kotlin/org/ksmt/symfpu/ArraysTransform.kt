package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.symfpu.SymFPUModel.Companion.declContainsFp
import org.ksmt.utils.cast

class ArraysTransform(val ctx: KContext) {

    val mapFpToBvDeclImpl = mutableMapOf<KDecl<*>, KConst<*>>()

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
    ): List<KDecl<*>> = decls.map {
        transformDecl(it)
    }


    fun transformDecl(it: KDecl<*>) = with(ctx) {
        val sort = it.sort
        when {
            !declContainsFp(it) -> {
                it
            }

            it is KConstDecl<*> -> {
                val newSort = transformSortRemoveFP(sort)
                mapFpToBvDeclImpl.getOrPut(it) {
                    mkConst(it.name + "!tobv!", newSort).cast()
                }.decl
            }

            // todo
            it is KFuncDecl -> {
                mkFreshFuncDecl(it.name, transformSortRemoveFP(it.sort), it.argSorts.fpToBvSorts())
            }

            else -> throw IllegalStateException("Unexpected decl type: $it")
        }
    }

    private fun List<KSort>.fpToBvSorts() = map { ctx.transformSortRemoveFP(it) }


    fun <R : KSort> arraySelectUnpacked(sort: R, res: KExpr<R>): KExpr<R> = with(ctx) {
        if (sort is KFpSort) {
            val resTyped: KExpr<KBvSort> = res.cast()
            unpack(sort, resTyped).cast()
        } else {
            res
        }
    }

    companion object {
        internal fun <D : KSort> packToBvIfUnpacked(expr: KExpr<D>): KExpr<KSort> = with(expr.ctx) {
            ((expr as? UnpackedFp<*>)?.let { packToBv(expr) } ?: expr).cast()
        }

        private fun KContext.mkAnyArraySort(domain: List<KSort>, range: KSort): KArraySortBase<KSort> =
            mkAnyArrayOperation(
                domain,
                { d0 -> mkArraySort(d0, range) },
                { d0, d1 -> mkArraySort(d0, d1, range) },
                { d0, d1, d2 -> mkArraySort(d0, d1, d2, range) },
                { mkArrayNSort(it, range) }
            )

        fun <A : KArraySortBase<*>> KContext.mkAnyArraySelect(
            array: KExpr<A>,
            indices: List<KExpr<KSort>>,
        ): KExpr<KSort> {
            val domain = array.sort.domainSorts
            return when (domain.size) {
                KArraySort.DOMAIN_SIZE -> mkArraySelect(array.cast(), indices.single())
                KArray2Sort.DOMAIN_SIZE -> mkArraySelect(array.cast(), indices.first(), indices.last())
                KArray3Sort.DOMAIN_SIZE -> {
                    val (d0, d1, d2) = indices
                    mkArraySelect(array.cast(), d0, d1, d2)
                }

                else -> mkArrayNSelect(array.cast(), indices)
            }
        }

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
                KArraySort.DOMAIN_SIZE -> mkArrayStore(array.cast(), indices.single(), value)
                KArray2Sort.DOMAIN_SIZE -> mkArrayStore(array.cast(), indices.first(), indices.last(), value)
                KArray3Sort.DOMAIN_SIZE -> {
                    val (d0, d1, d2) = indices
                    mkArrayStore(array.cast(), d0, d1, d2, value)
                }
                else -> mkArrayNStore(array.cast(), indices, value)
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


        fun transformedArraySort(
            expr: KExpr<KArraySortBase<*>>,
        ): KArraySortBase<KSort> = with(expr.ctx) {
            return transformArraySort(expr.sort)
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


