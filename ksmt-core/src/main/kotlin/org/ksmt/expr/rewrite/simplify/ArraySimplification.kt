package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

fun <D : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>,
    value: KExpr<R>
) : KExpr<KArraySort<D, R>> {
    // (store (const v) i v) ==> (const v)
    if (array is KArrayConst<D, R> && array.value == value) {
        return array
    }

    // (store a i (select a i)) ==> a
    if (value is KArraySelect<*, *> && array == value.array && index == value.index) {
        return array
    }

    return mkArrayStoreNoSimplify(array, index, value)
}

fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    value: KExpr<R>
): KExpr<KArray2Sort<D0, D1, R>> = TODO()

fun <D0 : KSort, D1 : KSort, D2: KSort, R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>,
    value: KExpr<R>
): KExpr<KArray3Sort<D0, D1, D2, R>> = TODO()

fun <R : KSort> KContext.simplifyArrayStore(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>,
    value: KExpr<R>
): KExpr<KArrayNSort<R>> = TODO()

fun <D : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArraySort<D, R>>,
    index: KExpr<D>
): KExpr<R> {
    var currentArray = array

    while (currentArray is KArrayStore<D, R>) {
        // (select (store i v) i) ==> v
        if (currentArray.index == index) {
            return currentArray.value
        }

        // (select (store a i v) j), i != j ==> (select a j)
        if (index is KInterpretedValue<D> && currentArray.index is KInterpretedValue<D>) {
            currentArray = currentArray.array
        } else {
            // possibly equal index, we can't expand stores
            break
        }
    }

    when (currentArray) {
        // (select (const v) i) ==> v
        is KArrayConst<D, R> -> {
            return currentArray.value
        }
        // (select (lambda x body) i) ==> body[i/x]
        is KArrayLambda<D, R> -> {
            val resolvedBody = KExprSubstitutor(this).apply {
                val indexVarExpr = mkConstApp(currentArray.indexVarDecl)
                substitute(indexVarExpr, index)
            }.apply(currentArray.body)
            return resolvedBody
        }
    }

    return mkArraySelectNoSimplify(currentArray, index)
}

fun <D0 : KSort, D1 : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArray2Sort<D0, D1, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>
): KExpr<R> = TODO()

fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    index0: KExpr<D0>,
    index1: KExpr<D1>,
    index2: KExpr<D2>
): KExpr<R> = TODO()

fun <R : KSort> KContext.simplifyArraySelect(
    array: KExpr<KArrayNSort<R>>,
    indices: List<KExpr<*>>
): KExpr<R> = TODO()
