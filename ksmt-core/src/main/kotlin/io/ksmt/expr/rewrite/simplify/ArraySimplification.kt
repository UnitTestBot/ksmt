package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

fun <D : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    value: KExpr<R>
): KExpr<KArraySort<D, R>> = simplifyArrayStoreLight(array, index, value, ::mkArrayStoreNoSimplify)

fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    value: KExpr<R>
): KExpr<KArray2Sort<D0, D1, R>> = simplifyArrayStoreLight(array, index0, index1, value, ::mkArrayStoreNoSimplify)

fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>,
    value: KExpr<R>
): KExpr<KArray3Sort<D0, D1, D2, R>> =
    simplifyArrayStoreLight(array, index0, index1, index2, value, ::mkArrayStoreNoSimplify)

fun <R : KSort> KContext.simplifyArrayNStore(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>,
    value: KExpr<R>
): KExpr<KArrayNSort<R>> = simplifyArrayNStoreLight(array, indices, value, ::mkArrayNStoreNoSimplify)

fun <D : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>
): KExpr<R> = simplifySelectFromArrayStore(
    array = array,
    index = index,
    storeIndexMatch = { store: KArrayStore<D, R>, idx: KExpr<D> -> idx == store.index },
    storeIndexDistinct = { store: KArrayStore<D, R>, idx: KExpr<D> -> areDistinct(idx, store.index) }
) { array2, i ->
    simplifyArraySelectLambda(array2, i, cont = ::mkArraySelectNoSimplify)
}

fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>
): KExpr<R> = simplifySelectFromArrayStore(
    array = array,
    index0 = index0,
    index1 = index1,
    storeIndexMatch = { store: KArray2Store<D0, D1, R>, idx0: KExpr<D0>, idx1: KExpr<D1> ->
        idx0 == store.index0 && idx1 == store.index1
    },
    storeIndexDistinct = { store: KArray2Store<D0, D1, R>, idx0: KExpr<D0>, idx1: KExpr<D1> ->
        areDistinct(idx0, store.index0) || areDistinct(idx1, store.index1)
    }
) { array2, i0, i1 ->
    simplifyArraySelectLambda(array2, i0, i1, cont = ::mkArraySelectNoSimplify)
}

fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>
): KExpr<R> = simplifySelectFromArrayStore(
    array = array,
    index0 = index0,
    index1 = index1,
    index2 = index2,
    storeIndexMatch = { store: KArray3Store<D0, D1, D2, R>, idx0: KExpr<D0>, idx1: KExpr<D1>, idx2: KExpr<D2> ->
        idx0 == store.index0 && idx1 == store.index1 && idx2 == store.index2
    },
    storeIndexDistinct = { store: KArray3Store<D0, D1, D2, R>, idx0: KExpr<D0>, idx1: KExpr<D1>, idx2: KExpr<D2> ->
        areDistinct(idx0, store.index0)
                || areDistinct(idx1, store.index1)
                || areDistinct(idx2, store.index2)
    }
) { array2, i0, i1, i2 ->
    simplifyArraySelectLambda(array2, i0, i1, i2, cont = ::mkArraySelectNoSimplify)
}

fun <R : KSort> KContext.simplifyArrayNSelect(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>
): KExpr<R> = simplifyArrayNSelectFromArrayStore(
    array = array,
    indices = indices,
    storeIndexMatch = { store: KArrayNStore<R>, idxs: List<KExpr<*>> -> store.indices == idxs },
    storeIndexDistinct = { store: KArrayNStore<R>, idxs: List<KExpr<*>> ->
        idxs.zip(store.indices).any { areDistinct(it.first.uncheckedCast(), it.second) }
    }
) { array2, indices2 ->
    simplifyArrayNSelectLambda(array2, indices2, cont = ::mkArrayNSelectNoSimplify)
}

private fun <T : KSort> areDistinct(left: KExpr<T>, right: KExpr<T>): Boolean =
    left is KInterpretedValue<T> && right is KInterpretedValue<T> && left != right
