package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KArray2Lambda
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Lambda
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambda
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayNLambda
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArraySelectBase
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KExpr
import io.ksmt.expr.rewrite.KExprSubstitutor
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

inline fun <
    reified A : KArraySortBase<R>, R : KSort,
    reified S : KArraySelectBase<out A, R>
> simplifyArrayStoreLight(
    array: KExpr<A>,
    value: KExpr<R>,
    selectIndicesMatch: (S) -> Boolean,
    default: () -> KExpr<A>
): KExpr<A> {
    // (store (const v) i v) ==> (const v)
    if (array is KArrayConst<A, *> && array.value == value) {
        return array
    }

    // (store a i (select a i)) ==> a
    if (value is S && array == value.array && selectIndicesMatch(value)) {
        return array
    }

    return default()
}

inline fun <D : KSort, R : KSort> KContext.simplifyArrayStoreLight(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    value: KExpr<R>,
    cont: (KExpr<KArraySort<D, R>>, KExpr<D>, KExpr<R>) -> KExpr<KArraySort<D, R>>
): KExpr<KArraySort<D, R>> = simplifyArrayStoreLight(
    array,
    value,
    selectIndicesMatch = { select: KArraySelect<D, R> -> index == select.index },
    default = { cont(array, index, value) }
)

inline fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArrayStoreLight(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    value: KExpr<R>,
    cont: (KExpr<KArray2Sort<D0, D1, R>>, KExpr<D0>, KExpr<D1>, KExpr<R>) -> KExpr<KArray2Sort<D0, D1, R>>
): KExpr<KArray2Sort<D0, D1, R>> = simplifyArrayStoreLight(
    array,
    value,
    selectIndicesMatch = { select: KArray2Select<D0, D1, R> -> index0 == select.index0 && index1 == select.index1 },
    default = { cont(array, index0, index1, value) }
)

@Suppress("LongParameterList")
inline fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArrayStoreLight(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>,
    value: KExpr<R>,
    cont: (
        KExpr<KArray3Sort<D0, D1, D2, R>>,
        KExpr<D0>,
        KExpr<D1>,
        KExpr<D2>,
        KExpr<R>
    ) -> KExpr<KArray3Sort<D0, D1, D2, R>>
): KExpr<KArray3Sort<D0, D1, D2, R>> = simplifyArrayStoreLight(
    array,
    value,
    selectIndicesMatch = { select: KArray3Select<D0, D1, D2, R> ->
        index0 == select.index0 && index1 == select.index1 && index2 == select.index2
    },
    default = { cont(array, index0, index1, index2, value) }
)

inline fun <R : KSort> KContext.simplifyArrayNStoreLight(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>,
    value: KExpr<R>,
    cont: (KExpr<KArrayNSort<R>>, List<KExpr<*>>, KExpr<R>) -> KExpr<KArrayNSort<R>>
): KExpr<KArrayNSort<R>> = simplifyArrayStoreLight(
    array,
    value,
    selectIndicesMatch = { select: KArrayNSelect<R> -> indices == select.indices },
    default = { cont(array, indices, value) }
)

@Suppress("LoopWithTooManyJumpStatements")
/**
 * Simplify select from a chain of array store expressions.
 *
 * If array stores are not analyzed (see [KArrayStoreBase.analyzeStore])
 * this operation will traverse whole array store chain.
 * Otherwise, we speed up the traversal with [KArrayStoreBase.findArrayToSelectFrom].
 * In the case of a one-dimensional arrays, this operation is guaranteed
 * to perform only one iteration of the loop (constant).
 * For the multi-dimensional arrays, usually only a few iterations will be performed,
 * but in the worst case we may traverse the entire stores chain.
 * */
inline fun <
    reified A : KArraySortBase<R>,
    R : KSort,
    reified S : KArrayStoreBase<A, R>
> KContext.simplifySelectFromArrayStore(
    initialArray: KExpr<A>,
    storeIndicesMatch: (S) -> Boolean,
    storeIndicesDistinct: (S) -> Boolean,
    findArrayToSelectFrom: (S) -> KExpr<A>,
    default: (KExpr<A>) -> KExpr<R>
): KExpr<R> {
    var array = initialArray
    while (array is S) {
        // Try fast index lookup
        array = findArrayToSelectFrom(array)

        if (array !is S) continue

        // (select (store i v) i) ==> v
        if (storeIndicesMatch(array)) {
            return array.value
        }

        // (select (store a i v) j), i != j ==> (select a j)
        if (storeIndicesDistinct(array)) {
            array = array.array
        } else {
            // possibly equal index, we can't expand stores
            break
        }
    }

    return if (array is KArrayConst<A, *>) {
        array.value.uncheckedCast()
    } else {
        default(array)
    }
}

inline fun <D : KSort, R : KSort> KContext.simplifySelectFromArrayStore(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    storeIndexMatch: KContext.(KArrayStore<D, R>, KExpr<D>) -> Boolean,
    storeIndexDistinct: KContext.(KArrayStore<D, R>, KExpr<D>) -> Boolean,
    cont: (KExpr<KArraySort<D, R>>, KExpr<D>) -> KExpr<R>
): KExpr<R> =
    simplifySelectFromArrayStore(
        initialArray = array,
        storeIndicesMatch = { store: KArrayStore<D, R> -> storeIndexMatch(store, index) },
        storeIndicesDistinct = { store: KArrayStore<D, R> -> storeIndexDistinct(store, index)},
        findArrayToSelectFrom = { store: KArrayStore<D, R> -> store.findArrayToSelectFrom(index) },
        default = { cont(it, index) }
    )

@Suppress("LongParameterList")
inline fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifySelectFromArrayStore(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    storeIndexMatch: KContext.(KArray2Store<D0, D1, R>, KExpr<D0>, KExpr<D1>) -> Boolean,
    storeIndexDistinct: KContext.(KArray2Store<D0, D1, R>, KExpr<D0>, KExpr<D1>) -> Boolean,
    cont: (KExpr<KArray2Sort<D0, D1, R>>, KExpr<D0>, KExpr<D1>) -> KExpr<R>
): KExpr<R> =
    simplifySelectFromArrayStore(
        initialArray = array,
        storeIndicesMatch = { store: KArray2Store<D0, D1, R> -> storeIndexMatch(store, index0, index1) },
        storeIndicesDistinct = { store: KArray2Store<D0, D1, R> -> storeIndexDistinct(store, index0, index1) },
        findArrayToSelectFrom = { store: KArray2Store<D0, D1, R> ->
            store.findArrayToSelectFrom(index0, index1)
        },
        default = { cont(it, index0, index1) }
    )

@Suppress("LongParameterList")
inline fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifySelectFromArrayStore(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>,
    storeIndexMatch: KContext.(KArray3Store<D0, D1, D2, R>, KExpr<D0>, KExpr<D1>, KExpr<D2>) -> Boolean,
    storeIndexDistinct: KContext.(KArray3Store<D0, D1, D2, R>, KExpr<D0>, KExpr<D1>, KExpr<D2>) -> Boolean,
    cont: (KExpr<KArray3Sort<D0, D1, D2, R>>, KExpr<D0>, KExpr<D1>, KExpr<D2>) -> KExpr<R>
): KExpr<R> =
    simplifySelectFromArrayStore(
        initialArray = array,
        storeIndicesMatch = { store: KArray3Store<D0, D1, D2, R> -> storeIndexMatch(store, index0, index1, index2) },
        storeIndicesDistinct = { store: KArray3Store<D0, D1, D2, R> ->
            storeIndexDistinct(store, index0, index1, index2)
                               },
        findArrayToSelectFrom = { s: KArray3Store<D0, D1, D2, R> ->
            s.findArrayToSelectFrom(index0, index1, index2)
        },
        default = { cont(it, index0, index1, index2) }
    )

inline fun <R : KSort> KContext.simplifyArrayNSelectFromArrayStore(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>,
    storeIndexMatch: KContext.(KArrayNStore<R>, List<KExpr<*>>) -> Boolean,
    storeIndexDistinct: KContext.(KArrayNStore<R>, List<KExpr<*>>) -> Boolean,
    cont: (KExpr<KArrayNSort<R>>, List<KExpr<*>>) -> KExpr<R>
): KExpr<R> =
    simplifySelectFromArrayStore(
        initialArray = array,
        storeIndicesMatch = { store: KArrayNStore<R> -> storeIndexMatch(store, indices) },
        storeIndicesDistinct = { store: KArrayNStore<R> -> storeIndexDistinct(store, indices) },
        findArrayToSelectFrom = { store: KArrayNStore<R> -> store.findArrayToSelectFrom(indices) },
        default = { cont(it, indices) }
    )

@Suppress("LongParameterList")
inline fun <
    reified A : KArraySortBase<R>,
    R : KSort,
    reified L : KArrayLambdaBase<A, R>
> KContext.simplifyArraySelectLambda(
    array: KExpr<A>,
    mkLambdaSubstitution: KExprSubstitutor.(L) -> Unit,
    rewriteBody: KContext.(KExpr<R>) -> KExpr<R>,
    default: (KExpr<A>) -> KExpr<R>
): KExpr<R> {
    if (array is L) {
        val resolvedBody = KExprSubstitutor(this).apply {
            mkLambdaSubstitution(array)
        }.apply(array.body)
        return rewriteBody(resolvedBody)
    }

    return default(array)
}

inline fun <D : KSort, R : KSort> KContext.simplifyArraySelectLambda(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    rewriteBody: KContext.(KExpr<R>) -> KExpr<R> = { it },
    cont: (KExpr<KArraySort<D, R>>, KExpr<D>) -> KExpr<R>
): KExpr<R> = simplifyArraySelectLambda(
    array = array,
    mkLambdaSubstitution = { lambda: KArrayLambda<D, R> ->
        substitute(mkConstApp(lambda.indexVarDecl), index)
    },
    rewriteBody = rewriteBody,
    default = { cont(it, index) }
)

inline fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArraySelectLambda(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    rewriteBody: KContext.(KExpr<R>) -> KExpr<R> = { it },
    cont: (KExpr<KArray2Sort<D0, D1, R>>, KExpr<D0>, KExpr<D1>) -> KExpr<R>
): KExpr<R> = simplifyArraySelectLambda(
    array = array,
    mkLambdaSubstitution = { lambda: KArray2Lambda<D0, D1, R> ->
        substitute(mkConstApp(lambda.indexVar0Decl), index0)
        substitute(mkConstApp(lambda.indexVar1Decl), index1)
    },
    rewriteBody = rewriteBody,
    default = { cont(it, index0, index1) }
)

@Suppress("LongParameterList")
inline fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArraySelectLambda(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>,
    rewriteBody: KContext.(KExpr<R>) -> KExpr<R> = { it },
    cont: (KExpr<KArray3Sort<D0, D1, D2, R>>, KExpr<D0>, KExpr<D1>, KExpr<D2>) -> KExpr<R>
): KExpr<R> = simplifyArraySelectLambda(
    array = array,
    mkLambdaSubstitution = { lambda: KArray3Lambda<D0, D1, D2, R> ->
        substitute(mkConstApp(lambda.indexVar0Decl), index0)
        substitute(mkConstApp(lambda.indexVar1Decl), index1)
        substitute(mkConstApp(lambda.indexVar2Decl), index2)
    },
    rewriteBody = rewriteBody,
    default = { cont(it, index0, index1, index2) }
)

inline fun <R : KSort> KContext.simplifyArrayNSelectLambda(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>,
    rewriteBody: KContext.(KExpr<R>) -> KExpr<R> = { it },
    cont: (KExpr<KArrayNSort<R>>, List<KExpr<*>>) -> KExpr<R>
): KExpr<R> = simplifyArraySelectLambda(
    array = array,
    mkLambdaSubstitution = { lambda: KArrayNLambda<R> ->
        lambda.indexVarDeclarations.zip(indices) { varDecl, index ->
            substitute(mkConstApp(varDecl).uncheckedCast<_, KExpr<KSort>>(), index.uncheckedCast())
        }
    },
    rewriteBody = rewriteBody,
    default = { cont(it, indices) }
)

