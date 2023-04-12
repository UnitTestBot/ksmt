package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KArray2Lambda
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Lambda
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayLambdaBase
import org.ksmt.expr.KArrayNLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArraySelectBase
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KArrayStoreBase
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

private inline fun <
    reified A : KArraySortBase<R>, R : KSort,
    reified S : KArraySelectBase<out A, R>
> simplifyArrayStore(
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

fun <D : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    value: KExpr<R>
): KExpr<KArraySort<D, R>> = simplifyArrayStore(
    array,
    value,
    selectIndicesMatch = { select: KArraySelect<D, R> -> index == select.index },
    default = { mkArrayStoreNoSimplify(array, index, value) }
)

fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    value: KExpr<R>
): KExpr<KArray2Sort<D0, D1, R>> = simplifyArrayStore(
    array,
    value,
    selectIndicesMatch = { select: KArray2Select<D0, D1, R> -> index0 == select.index0 && index1 == select.index1 },
    default = { mkArrayStoreNoSimplify(array, index0, index1, value) }
)

fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>,
    value: KExpr<R>
): KExpr<KArray3Sort<D0, D1, D2, R>> = simplifyArrayStore(
    array,
    value,
    selectIndicesMatch = { select: KArray3Select<D0, D1, D2, R> ->
        index0 == select.index0 && index1 == select.index1 && index2 == select.index2
    },
    default = { mkArrayStoreNoSimplify(array, index0, index1, index2, value) }
)

fun <R : KSort> KContext.simplifyArrayNStore(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>,
    value: KExpr<R>
): KExpr<KArrayNSort<R>> = simplifyArrayStore(
    array,
    value,
    selectIndicesMatch = { select: KArrayNSelect<R> -> indices == select.indices },
    default = { mkArrayNStoreNoSimplify(array, indices, value) }
)

@Suppress("LongParameterList")
private inline fun <
    reified A : KArraySortBase<R>,
    R : KSort,
    reified S : KArrayStoreBase<A, R>,
    reified L : KArrayLambdaBase<A, R>
> KContext.simplifyArraySelect(
    array: KExpr<A>,
    storeIndicesMatch: (S) -> Boolean,
    storeIndicesDistinct: (S) -> Boolean,
    findArrayToSelectFrom: (S) -> KExpr<A>,
    mkLambdaSubstitution: KExprSubstitutor.(L) -> Unit,
    default: (KExpr<A>) -> KExpr<R>
): KExpr<R> = simplifySelectFromArrayStore<A, R, S>(
    initialArray = array,
    storeIndicesMatch = { storeIndicesMatch(it) },
    storeIndicesDistinct = { storeIndicesDistinct(it) },
    findArrayToSelectFrom = { findArrayToSelectFrom(it) }
) { currentArray ->
    when (currentArray) {
        // (select (const v) i) ==> v
        is KArrayConst<A, *> -> {
            return currentArray.value.uncheckedCast()
        }
        // (select (lambda x body) i) ==> body[i/x]
        is L -> {
            val resolvedBody = KExprSubstitutor(this).apply {
                mkLambdaSubstitution(currentArray)
            }.apply(currentArray.body)
            return resolvedBody
        }

        else -> default(currentArray)
    }
}

fun <D : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>
): KExpr<R> = simplifyArraySelect(
    array = array,
    storeIndicesMatch = { store: KArrayStore<D, R> -> index == store.index },
    storeIndicesDistinct = { store: KArrayStore<D, R> -> areDistinct(index, store.index) },
    findArrayToSelectFrom = { store: KArrayStore<D, R> -> store.findArrayToSelectFrom(index) },
    mkLambdaSubstitution = { lambda: KArrayLambda<D, R> ->
        substitute(mkConstApp(lambda.indexVarDecl), index)
    },
    default = { mkArraySelectNoSimplify(it, index) }
)

fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>
): KExpr<R> = simplifyArraySelect(
    array = array,
    storeIndicesMatch = { store: KArray2Store<D0, D1, R> ->
        index0 == store.index0 && index1 == store.index1
    },
    storeIndicesDistinct = { store: KArray2Store<D0, D1, R> ->
        areDistinct(index0, store.index0) || areDistinct(index1, store.index1)
    },
    findArrayToSelectFrom = { store: KArray2Store<D0, D1, R> ->
        store.findArrayToSelectFrom(index0, index1)
    },
    mkLambdaSubstitution = { lambda: KArray2Lambda<D0, D1, R> ->
        substitute(mkConstApp(lambda.indexVar0Decl), index0)
        substitute(mkConstApp(lambda.indexVar1Decl), index1)
    },
    default = { mkArraySelectNoSimplify(it, index0, index1) }
)

fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>
): KExpr<R> = simplifyArraySelect(
    array = array,
    storeIndicesMatch = { store: KArray3Store<D0, D1, D2, R> ->
        index0 == store.index0 && index1 == store.index1 && index2 == store.index2
    },
    storeIndicesDistinct = { store: KArray3Store<D0, D1, D2, R> ->
        areDistinct(index0, store.index0)
                || areDistinct(index1, store.index1)
                || areDistinct(index2, store.index2)
    },
    findArrayToSelectFrom = { s: KArray3Store<D0, D1, D2, R> ->
        s.findArrayToSelectFrom(index0, index1, index2)
    },
    mkLambdaSubstitution = { lambda: KArray3Lambda<D0, D1, D2, R> ->
        substitute(mkConstApp(lambda.indexVar0Decl), index0)
        substitute(mkConstApp(lambda.indexVar1Decl), index1)
        substitute(mkConstApp(lambda.indexVar2Decl), index2)
    },
    default = { mkArraySelectNoSimplify(it, index0, index1, index2) }
)

fun <R : KSort> KContext.simplifyArrayNSelect(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>
): KExpr<R> = simplifyArraySelect(
    array = array,
    storeIndicesMatch = { store: KArrayNStore<R> -> indices == store.indices },
    storeIndicesDistinct = { store: KArrayNStore<R> ->
        indices.zip(store.indices).any { areDistinct(it.first.uncheckedCast(), it.second) }
    },
    findArrayToSelectFrom = { store: KArrayNStore<R> -> store.findArrayToSelectFrom(indices) },
    mkLambdaSubstitution = { lambda: KArrayNLambda<R> ->
        lambda.indexVarDeclarations.zip(indices) { varDecl, index ->
            substitute(mkConstApp(varDecl).uncheckedCast<_, KExpr<KSort>>(), index.uncheckedCast())
        }
    },
    default = { mkArrayNSelectNoSimplify(it, indices) }
)

@Suppress("LoopWithTooManyJumpStatements")
/**
 * Simplify select from a chain on array store expressions.
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
    A : KArraySortBase<R>,
    R : KSort,
    reified S : KArrayStoreBase<A, R>
> simplifySelectFromArrayStore(
    initialArray: KExpr<A>,
    storeIndicesMatch: (S) -> Boolean,
    storeIndicesDistinct: (S) -> Boolean,
    findArrayToSelectFrom: (S) -> KExpr<A>,
    simplifyResultArray: (KExpr<A>) -> KExpr<R>
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

    return simplifyResultArray(array)
}

private fun <T : KSort> areDistinct(left: KExpr<T>, right: KExpr<T>): Boolean =
    left is KInterpretedValue<T> && right is KInterpretedValue<T> && left != right
