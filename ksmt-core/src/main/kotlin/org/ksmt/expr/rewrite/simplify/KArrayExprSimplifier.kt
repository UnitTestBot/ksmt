package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KArray2Select
import org.ksmt.expr.KArray2Store
import org.ksmt.expr.KArray3Select
import org.ksmt.expr.KArray3Store
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArrayNSelect
import org.ksmt.expr.KArrayNStore
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArraySelectBase
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

interface KArrayExprSimplifier : KExprSimplifierBase {

    fun <A : KArraySortBase<R>, R : KSort> simplifyEqArray(
        lhs: KExpr<A>,
        rhs: KExpr<A>
    ): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        val leftArray = flatStoresGeneric(lhs.sort, lhs)
        val rightArray = flatStoresGeneric(rhs.sort, rhs)
        val lBase = leftArray.base
        val rBase = rightArray.base

        /**
         * (= (store a i v) (store b x y)) ==>
         * (and
         *   (= (select (store a i v) i) (select (store b x y) i))
         *   (= (select (store a i v) x) (select (store b x y) x))
         *   (= a b)
         * )
         */
        if (lBase == rBase || lBase is KArrayConst<*, *> && rBase is KArrayConst<*, *>) {
            val simplifiedExpr = simplifyArrayStoreEq(lhs, leftArray, rhs, rightArray)
            return rewrite(simplifiedExpr)
        }

        return mkEqNoSimplify(lhs, rhs)
    }

    /**
     * (= (store a i v) (store b x y)) ==>
     * (and
     *   (= (select (store a i v) i) (select (store b x y) i))
     *   (= (select (store a i v) x) (select (store b x y) x))
     *   (= a b)
     * )
     */
    private fun <A : KArraySortBase<R>, R : KSort> simplifyArrayStoreEq(
        lhs: KExpr<A>,
        leftArray: SimplifierFlatArrayStoreBaseExpr<A, R>,
        rhs: KExpr<A>,
        rightArray: SimplifierFlatArrayStoreBaseExpr<A, R>,
    ): SimplifierAuxExpression<KBoolSort> = auxExpr {
        val checks = arrayListOf<KExpr<KBoolSort>>()
        if (leftArray.base is KArrayConst<A, *> && rightArray.base is KArrayConst<A, *>) {
            // (= (const a) (const b)) ==> (= a b)
            checks += KEqExpr(
                ctx,
                leftArray.base.value.uncheckedCast<_, KExpr<KSort>>(),
                rightArray.base.value.uncheckedCast()
            )
        } else {
            check(leftArray.base == rightArray.base) {
                "Base arrays expected to be equal or constant"
            }
        }

        val leftArraySearchInfo = analyzeArrayStores(leftArray)
        val rightArraySearchInfo = analyzeArrayStores(rightArray)

        val allIndices = leftArraySearchInfo.storeIndices.keys + rightArraySearchInfo.storeIndices.keys
        for (idx in allIndices) {
            val lValue = selectArrayValue(idx, leftArraySearchInfo, lhs, leftArray)
            val rValue = selectArrayValue(idx, rightArraySearchInfo, rhs, rightArray)
            if (lValue != rValue) {
                checks += KEqExpr(ctx, lValue, rValue)
            }
        }

        KAndExpr(ctx, checks)
    }

    private class ArrayStoreSearchInfo<A : KArraySortBase<R>, R : KSort>(
        val storeIndices: Map<List<KExpr<*>>, Int>,
        val storeValues: List<KExpr<R>>,
        val nonConstants: List<List<KExpr<*>>>,
        val nonConstantsToCheck: IntArray
    )

    private fun <A : KArraySortBase<R>, R : KSort> selectArrayValue(
        selectIndex: List<KExpr<*>>,
        arraySearchInfo: ArrayStoreSearchInfo<A, R>,
        originalArrayExpr: KExpr<A>,
        flatArray: SimplifierFlatArrayStoreBaseExpr<A, R>
    ): KExpr<R> =
        findStoredArrayValue(arraySearchInfo, selectIndex)
            ?: flatArray.selectValue(originalArrayExpr, selectIndex)

    /**
     * Preprocess flat array stores for faster index search.
     * @see [findStoredArrayValue]
     * */
    private fun <A : KArraySortBase<R>, R : KSort> analyzeArrayStores(
        array: SimplifierFlatArrayStoreBaseExpr<A, R>
    ): ArrayStoreSearchInfo<A, R> {
        val indexId = hashMapOf<List<KExpr<*>>, Int>()
        val nonConstants = arrayListOf<List<KExpr<*>>>()
        val nonConstantsToCheck = IntArray(array.numIndices)

        for (i in 0 until array.numIndices) {
            val storeIndex = array.getStoreIndex(i)
            if (storeIndex !in indexId) {
                indexId[storeIndex] = i
                if (!storeIndex.all { it.definitelyIsConstant }) {
                    nonConstants += storeIndex
                }
            }
            nonConstantsToCheck[i] = nonConstants.size
        }

        return ArrayStoreSearchInfo(
            storeIndices = indexId,
            storeValues = array.values,
            nonConstants = nonConstants,
            nonConstantsToCheck = nonConstantsToCheck
        )
    }

    /**
     * Try to find stored value for the provided index.
     * @return null if there is no such index in the array,
     * or it is impossible to establish equality of some of the stored indices.
     * */
    private fun <A : KArraySortBase<R>, R : KSort> findStoredArrayValue(
        array: ArrayStoreSearchInfo<A, R>,
        selectIndex: List<KExpr<*>>,
    ): KExpr<R>? {
        val storeIndex = array.storeIndices[selectIndex] ?: return null

        /**
         * Since all constants are trivially comparable we need to check
         * only non-constant stored indices.
         * */
        val lastNonConstantToCheck = array.nonConstantsToCheck[storeIndex]
        for (i in 0 until lastNonConstantToCheck) {
            val nonConstant = array.nonConstants[i]
            if (!areDefinitelyDistinct(selectIndex, nonConstant)) return null
        }

        return array.storeValues[storeIndex]
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatStores1(expr) }
        )

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatStores2(expr) }
        )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatStores3(expr) }
        )

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        simplifyExpr(
            expr = expr,
            preprocess = { flatStoresN(expr) }
        )

    private fun <D : KSort, R : KSort> transform(expr: SimplifierFlatArrayStoreExpr<D, R>): KExpr<KArraySort<D, R>> =
        simplifyExpr(expr, expr.unwrap()) { args: List<KExpr<KSort>> ->
            simplifyArrayStore(expr.wrap(args))
        }

    private fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: SimplifierFlatArray2StoreExpr<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        simplifyExpr(expr, expr.unwrap()) { args: List<KExpr<KSort>> ->
            simplifyArrayStore(expr.wrap(args))
        }

    private fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: SimplifierFlatArray3StoreExpr<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        simplifyExpr(expr, expr.unwrap()) { args: List<KExpr<KSort>> ->
            simplifyArrayStore(expr.wrap(args))
        }

    private fun <R : KSort> transform(expr: SimplifierFlatArrayNStoreExpr<R>): KExpr<KArrayNSort<R>> =
        simplifyExpr(expr, expr.unwrap()) { args: List<KExpr<KSort>> ->
            simplifyArrayStore(expr.wrap(args))
        }

    private fun <D : KSort, R : KSort> simplifyArrayStore(
        expr: SimplifierFlatArrayStoreExpr<D, R>
    ): KExpr<KArraySort<D, R>> {
        val indices = expr.indices
        val storedIndices = hashMapOf<KExpr<D>, Int>()
        val simplifiedIndices = arrayListOf<KExpr<D>>()

        return simplifyArrayStore(
            expr = expr,
            findStoredIndexPosition = { i ->
                storedIndices[indices[i]]
            },
            saveStoredIndexPosition = { i, indexPosition ->
                storedIndices[indices[i]] = indexPosition
            },
            indexIsConstant = { i ->
                indices[i].definitelyIsConstant
            },
            addSimplifiedIndex = { i ->
                simplifiedIndices.add(indices[i])
            },
            distinctWithSimplifiedIndex = { i, simplifiedIdx ->
                areDefinitelyDistinct(indices[i], simplifiedIndices[simplifiedIdx])
            },
            selectIndicesMatch = { select: KArraySelect<D, R>, i ->
                indices[i] == select.index
            },
            mkSimplifiedStore = { array, simplifiedIdx, value ->
                ctx.mkArrayStoreNoSimplify(array, simplifiedIndices[simplifiedIdx], value)
            }
        )
    }

    private fun <D0 : KSort, D1 : KSort, R : KSort> simplifyArrayStore(
        expr: SimplifierFlatArray2StoreExpr<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> {
        val indices0 = expr.indices0
        val indices1 = expr.indices1

        val storedIndices = hashMapOf<KExpr<D0>, MutableMap<KExpr<D1>, Int>>()
        val simplifiedIndices0 = arrayListOf<KExpr<D0>>()
        val simplifiedIndices1 = arrayListOf<KExpr<D1>>()

        return simplifyArrayStore(
            expr = expr,
            findStoredIndexPosition = { i ->
                storedIndices[indices0[i]]?.get(indices1[i])
            },
            saveStoredIndexPosition = { i, indexPosition ->
                val index1Map = storedIndices.getOrPut(indices0[i]) { hashMapOf() }
                index1Map[indices1[i]] = indexPosition
            },
            indexIsConstant = { i ->
                indices0[i].definitelyIsConstant && indices1[i].definitelyIsConstant
            },
            addSimplifiedIndex = { i ->
                simplifiedIndices0.add(indices0[i])
                simplifiedIndices1.add(indices1[i])
            },
            distinctWithSimplifiedIndex = { i, simplifiedIdx ->
                areDefinitelyDistinct(indices0[i], simplifiedIndices0[simplifiedIdx])
                    && areDefinitelyDistinct(indices1[i], simplifiedIndices1[simplifiedIdx])
            },
            selectIndicesMatch = { select: KArray2Select<D0, D1, R>, i ->
                indices0[i] == select.index0 && indices1[i] == select.index1
            },
            mkSimplifiedStore = { array, simplifiedIdx, value ->
                ctx.mkArrayStoreNoSimplify(
                    array,
                    simplifiedIndices0[simplifiedIdx],
                    simplifiedIndices1[simplifiedIdx],
                    value
                )
            }
        )
    }

    private fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> simplifyArrayStore(
        expr: SimplifierFlatArray3StoreExpr<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> {
        val indices0 = expr.indices0
        val indices1 = expr.indices1
        val indices2 = expr.indices2

        val storedIndices = hashMapOf<KExpr<D0>, MutableMap<KExpr<D1>, MutableMap<KExpr<D2>, Int>>>()
        val simplifiedIndices0 = arrayListOf<KExpr<D0>>()
        val simplifiedIndices1 = arrayListOf<KExpr<D1>>()
        val simplifiedIndices2 = arrayListOf<KExpr<D2>>()

        return simplifyArrayStore(
            expr = expr,
            findStoredIndexPosition = { i ->
                storedIndices[indices0[i]]?.get(indices1[i])?.get(indices2[i])
            },
            saveStoredIndexPosition = { i, indexPosition ->
                val index1Map = storedIndices.getOrPut(indices0[i]) { hashMapOf() }
                val index2Map = index1Map.getOrPut(indices1[i]) { hashMapOf() }
                index2Map[indices2[i]] = indexPosition
            },
            indexIsConstant = { i ->
                indices0[i].definitelyIsConstant
                    && indices1[i].definitelyIsConstant
                    && indices2[i].definitelyIsConstant
            },
            addSimplifiedIndex = { i ->
                simplifiedIndices0.add(indices0[i])
                simplifiedIndices1.add(indices1[i])
                simplifiedIndices2.add(indices2[i])
            },
            distinctWithSimplifiedIndex = { i, simplifiedIdx ->
                areDefinitelyDistinct(indices0[i], simplifiedIndices0[simplifiedIdx])
                    && areDefinitelyDistinct(indices1[i], simplifiedIndices1[simplifiedIdx])
                    && areDefinitelyDistinct(indices2[i], simplifiedIndices2[simplifiedIdx])
            },
            selectIndicesMatch = { select: KArray3Select<D0, D1, D2, R>, i ->
                indices0[i] == select.index0
                    && indices1[i] == select.index1
                    && indices2[i] == select.index2
            },
            mkSimplifiedStore = { array, simplifiedIdx, value ->
                ctx.mkArrayStoreNoSimplify(
                    array,
                    simplifiedIndices0[simplifiedIdx],
                    simplifiedIndices1[simplifiedIdx],
                    simplifiedIndices2[simplifiedIdx],
                    value
                )
            }
        )
    }

    private fun <R : KSort> simplifyArrayStore(
        expr: SimplifierFlatArrayNStoreExpr<R>
    ): KExpr<KArrayNSort<R>> {
        val indices = expr.indices
        val storedIndices = hashMapOf<List<KExpr<*>>, Int>()
        val simplifiedIndices = arrayListOf<List<KExpr<*>>>()

        return simplifyArrayStore(
            expr = expr,
            findStoredIndexPosition = { i ->
                storedIndices[indices[i]]
            },
            saveStoredIndexPosition = { i, indexPosition ->
                storedIndices[indices[i]] = indexPosition
            },
            indexIsConstant = { i ->
                indices[i].all { it.definitelyIsConstant }
            },
            addSimplifiedIndex = { i ->
                simplifiedIndices.add(indices[i])
            },
            distinctWithSimplifiedIndex = { i, simplifiedIdx ->
                areDefinitelyDistinct(indices[i], simplifiedIndices[simplifiedIdx])
            },
            selectIndicesMatch = { select: KArrayNSelect<R>, i ->
                indices[i] == select.indices
            },
            mkSimplifiedStore = { array, simplifiedIdx, value ->
                ctx.mkArrayStoreNoSimplify(array, simplifiedIndices[simplifiedIdx], value)
            }
        )
    }

    private inline fun <A : KArraySortBase<R>, R : KSort, reified S : KArraySelectBase<out A, R>> simplifyArrayStore(
        expr: SimplifierFlatArrayStoreBaseExpr<A, R>,
        findStoredIndexPosition: (Int) -> Int?,
        saveStoredIndexPosition: (Int, Int) -> Unit,
        indexIsConstant: (Int) -> Boolean,
        addSimplifiedIndex: (Int) -> Unit,
        distinctWithSimplifiedIndex: (Int, Int) -> Boolean,
        selectIndicesMatch: (S, Int) -> Boolean,
        mkSimplifiedStore: (KExpr<A>, Int, KExpr<R>) -> KExpr<A>
    ): KExpr<A> {
        if (expr.numIndices == 0) {
            return expr.base
        }

        val simplifiedValues = arrayListOf<KExpr<R>>()
        var numSimplifiedIndices = 0
        var lastNonConstantIndex = -1

        for (i in expr.numIndices - 1 downTo 0) {
            val value = expr.values[i]

            val storedIndexPosition = findStoredIndexPosition(i)
            if (storedIndexPosition != null) {

                /** Try push store.
                 * (store (store a i x) j y), i != j ==> (store (store a j y) i x)
                 *
                 * Store can be pushed only if all parent indices are definitely not equal.
                 * */
                val allParentIndicesAreDistinct = allParentSoreIndicesAreDistinct(
                    index = i,
                    storedIndexPosition = storedIndexPosition,
                    lastNonConstantIndex = lastNonConstantIndex,
                    numSimplifiedIndices = numSimplifiedIndices,
                    indexIsConstant = { x -> indexIsConstant(x) },
                    distinctWithSimplifiedIndex = { x, y -> distinctWithSimplifiedIndex(x, y) }
                )

                if (allParentIndicesAreDistinct) {
                    simplifiedValues[storedIndexPosition] = value
                    if (!indexIsConstant(i)) {
                        lastNonConstantIndex = maxOf(lastNonConstantIndex, storedIndexPosition)
                    }
                    continue
                }
            }

            // simplify direct store to array
            if (numSimplifiedIndices == 0) {
                // (store (const v) i v) ==> (const v)
                if (expr.base is KArrayConst<A, *> && expr.base.value == expr.values[i]) {
                    continue
                }

                // (store a i (select a i)) ==> a
                if (value is S && expr.base == value.array && selectIndicesMatch(value, i)) {
                    continue
                }
            }

            // store
            simplifiedValues.add(value)

            val indexPosition = numSimplifiedIndices++
            addSimplifiedIndex(i)
            saveStoredIndexPosition(i, indexPosition)

            if (!indexIsConstant(i)) {
                lastNonConstantIndex = maxOf(lastNonConstantIndex, indexPosition)
            }
        }

        var result = expr.base
        for (i in 0 until numSimplifiedIndices) {
            result = mkSimplifiedStore(result, i, simplifiedValues[i])
        }

        return result
    }

    private inline fun allParentSoreIndicesAreDistinct(
        index: Int,
        storedIndexPosition: Int,
        lastNonConstantIndex: Int,
        numSimplifiedIndices: Int,
        indexIsConstant: (Int) -> Boolean,
        distinctWithSimplifiedIndex: (Int, Int) -> Boolean
    ): Boolean {
        /**
         * Since all constants are trivially comparable we can guarantee, that
         * all parents are distinct.
         * Otherwise, we need to perform full check of parent indices.
         * */
        if (indexIsConstant(index) && storedIndexPosition >= lastNonConstantIndex) return true

        /**
         *  If non-constant index is among the indices we need to check,
         *  we can check it first. Since non-constant indices are usually
         *  not definitely distinct, we will not check all other indices.
         * */
        if (lastNonConstantIndex > storedIndexPosition
            && !distinctWithSimplifiedIndex(index, lastNonConstantIndex)
        ) {
            return false
        }

        for (checkIdx in (storedIndexPosition + 1) until numSimplifiedIndices) {
            if (!distinctWithSimplifiedIndex(index, checkIdx)) {
                // possibly equal index, we can't squash stores
                return false
            }
        }

        return true
    }

    fun <D : KSort, R : KSort> preprocessArraySelect(expr: KArraySelect<D, R>): SimplifierFlatArraySelectExpr<D, R> {
        val array = flatStores1(expr.array)
        return SimplifierFlatArraySelectExpr(
            ctx,
            original = expr.array,
            baseArray = array.base,
            storedValues = array.values,
            storedIndices = array.indices,
            index = expr.index
        )
    }

    fun <D0 : KSort, D1 : KSort, R : KSort> preprocessArraySelect(
        expr: KArray2Select<D0, D1, R>
    ): SimplifierFlatArray2SelectExpr<D0, D1, R> {
        val array = flatStores2(expr.array)
        return SimplifierFlatArray2SelectExpr(
            ctx,
            original = expr.array,
            baseArray = array.base,
            storedValues = array.values,
            storedIndices0 = array.indices0,
            storedIndices1 = array.indices1,
            index0 = expr.index0,
            index1 = expr.index1
        )
    }

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> preprocessArraySelect(
        expr: KArray3Select<D0, D1, D2, R>
    ): SimplifierFlatArray3SelectExpr<D0, D1, D2, R> {
        val array = flatStores3(expr.array)
        return SimplifierFlatArray3SelectExpr(
            ctx,
            original = expr.array,
            baseArray = array.base,
            storedValues = array.values,
            storedIndices0 = array.indices0,
            storedIndices1 = array.indices1,
            storedIndices2 = array.indices2,
            index0 = expr.index0,
            index1 = expr.index1,
            index2 = expr.index2
        )
    }

    fun <R : KSort> preprocessArraySelect(
        expr: KArrayNSelect<R>
    ): SimplifierFlatArrayNSelectExpr<R> {
        val array = flatStoresN(expr.array)
        return SimplifierFlatArrayNSelectExpr(
            ctx,
            original = expr.array,
            baseArray = array.base,
            storedValues = array.values,
            storedIndices = array.indices,
            indices = expr.indices
        )
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        simplifyExpr(
            expr = expr,
            preprocess = { preprocessArraySelect(expr) }
        )

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D0, D1, R>): KExpr<R> =
        simplifyExpr(
            expr = expr,
            preprocess = { preprocessArraySelect(expr) }
        )

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = simplifyExpr(
        expr = expr,
        preprocess = { preprocessArraySelect(expr) }
    )

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> =
        simplifyExpr(
            expr = expr,
            preprocess = { preprocessArraySelect(expr) }
        )

    /**
     * Try to simplify only indices first.
     * Usually this will be enough and we will not produce many irrelevant array store expressions.
     * */
    private fun <D : KSort, R : KSort> transform(expr: SimplifierFlatArraySelectExpr<D, R>): KExpr<R> =
        simplifyExpr(expr, listOf(expr.index) + expr.storedIndices) { allIndices ->
            val index = allIndices.first()
            val arrayIndices = allIndices.subList(fromIndex = 1, toIndex = allIndices.size)

            var arrayStoreIdx = 0
            while (arrayStoreIdx < arrayIndices.size) {
                val storeIdx = arrayIndices[arrayStoreIdx]

                // (select (store i v) i) ==> v
                if (storeIdx == index) {
                    return@simplifyExpr rewrite(expr.storedValues[arrayStoreIdx])
                }

                if (!areDefinitelyDistinct(index, storeIdx)) {
                    break
                }

                // (select (store a i v) j), i != j ==> (select a j)
                arrayStoreIdx++
            }

            if (arrayStoreIdx == arrayIndices.size) {
                return@simplifyExpr rewrite(SimplifierArraySelectExpr(ctx, expr.baseArray, index))
            }

            rewrite(SimplifierArraySelectExpr(ctx, expr.original, index))
        }

    private fun <D : KSort, R : KSort> transform(expr: SimplifierArraySelectExpr<D, R>): KExpr<R> =
        simplifyExpr(expr, expr.array, expr.index) { arrayArg, indexArg ->
            var array: KExpr<KArraySort<D, R>> = arrayArg.uncheckedCast()
            val index: KExpr<D> = indexArg.uncheckedCast()

            while (array is KArrayStore<D, R>) {
                // (select (store i v) i) ==> v
                if (array.index == index) {
                    return@simplifyExpr array.value
                }

                // (select (store a i v) j), i != j ==> (select a j)
                if (areDefinitelyDistinct(index, array.index)) {
                    array = array.array
                } else {
                    // possibly equal index, we can't expand stores
                    break
                }
            }

            when (array) {
                // (select (const v) i) ==> v
                is KArrayConst<D, R> -> {
                    array.value
                }
                // (select (lambda x body) i) ==> body[i/x]
                is KArrayLambda<D, R> -> {
                    val resolvedBody = KExprSubstitutor(ctx).apply {
                        val indexVarExpr = mkConstApp(array.indexVarDecl)
                        substitute(indexVarExpr, index)
                    }.apply(array.body)
                    rewrite(resolvedBody)
                }

                else -> {
                    mkArraySelectNoSimplify(array, index)
                }
            }
        }

    private fun <A : KArraySortBase<R>, R : KSort> flatStoresGeneric(
        sort: A, expr: KExpr<A>
    ): SimplifierFlatArrayStoreBaseExpr<A, R> = when (sort as KArraySortBase<R>) {
        is KArraySort<*, R> -> flatStores1<KSort, R>(expr.uncheckedCast()).uncheckedCast()
        is KArray2Sort<*, *, R> -> flatStores2<KSort, KSort, R>(expr.uncheckedCast()).uncheckedCast()
        is KArray3Sort<*, *, *, R> -> flatStores3<KSort, KSort, KSort, R>(expr.uncheckedCast()).uncheckedCast()
        is KArrayNSort<R> -> flatStoresN<R>(expr.uncheckedCast()).uncheckedCast()
    }

    private fun <D : KSort, R : KSort> flatStores1(
        expr: KExpr<KArraySort<D, R>>,
    ): SimplifierFlatArrayStoreExpr<D, R> {
        val indices = arrayListOf<KExpr<D>>()
        val values = arrayListOf<KExpr<R>>()
        var base = expr
        while (base is KArrayStore<D, R>) {
            indices.add(base.index)
            values.add(base.value)
            base = base.array
        }
        return SimplifierFlatArrayStoreExpr(
            ctx = ctx,
            original = expr,
            base = base,
            indices = indices,
            values = values
        )
    }

    private fun <D0 : KSort, D1 : KSort, R : KSort> flatStores2(
        expr: KExpr<KArray2Sort<D0, D1, R>>,
    ): SimplifierFlatArray2StoreExpr<D0, D1, R> {
        val indices0 = arrayListOf<KExpr<D0>>()
        val indices1 = arrayListOf<KExpr<D1>>()
        val values = arrayListOf<KExpr<R>>()
        var base = expr
        while (base is KArray2Store<D0, D1, R>) {
            indices0.add(base.index0)
            indices1.add(base.index1)
            values.add(base.value)
            base = base.array
        }
        return SimplifierFlatArray2StoreExpr(
            ctx = ctx,
            original = expr,
            base = base,
            indices0 = indices0,
            indices1 = indices1,
            values = values
        )
    }

    private fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> flatStores3(
        expr: KExpr<KArray3Sort<D0, D1, D2, R>>,
    ): SimplifierFlatArray3StoreExpr<D0, D1, D2, R> {
        val indices0 = arrayListOf<KExpr<D0>>()
        val indices1 = arrayListOf<KExpr<D1>>()
        val indices2 = arrayListOf<KExpr<D2>>()
        val values = arrayListOf<KExpr<R>>()
        var base = expr
        while (base is KArray3Store<D0, D1, D2, R>) {
            indices0.add(base.index0)
            indices1.add(base.index1)
            indices2.add(base.index2)
            values.add(base.value)
            base = base.array
        }
        return SimplifierFlatArray3StoreExpr(
            ctx = ctx,
            original = expr,
            base = base,
            indices0 = indices0,
            indices1 = indices1,
            indices2 = indices2,
            values = values
        )
    }

    private fun <R : KSort> flatStoresN(
        expr: KExpr<KArrayNSort<R>>,
    ): SimplifierFlatArrayNStoreExpr<R> {
        val indices = arrayListOf<List<KExpr<*>>>()
        val values = arrayListOf<KExpr<R>>()
        var base = expr
        while (base is KArrayNStore<R>) {
            indices.add(base.indices)
            values.add(base.value)
            base = base.array
        }
        return SimplifierFlatArrayNStoreExpr(
            ctx = ctx,
            original = expr,
            base = base,
            indices = indices,
            values = values
        )
    }

    private val KExpr<*>.definitelyIsConstant: Boolean
        get() = this is KInterpretedValue<*>

    private fun areDefinitelyDistinct(left: List<KExpr<*>>, right: List<KExpr<*>>): Boolean {
        for (i in left.indices) {
            val lhs: KExpr<KSort> = left[i].uncheckedCast()
            val rhs: KExpr<KSort> = right[i].uncheckedCast()
            if (!areDefinitelyDistinct(lhs, rhs)) return false
        }
        return true
    }

    /**
     * Auxiliary expression to handle expanded array stores.
     * @see [SimplifierAuxExpression]
     * */
    sealed class SimplifierFlatArrayStoreBaseExpr<A : KArraySortBase<R>, R : KSort>(
        ctx: KContext,
        val numIndices: Int,
        val original: KExpr<A>,
        val base: KExpr<A>,
        val values: List<KExpr<R>>
    ) : KExpr<A>(ctx) {
        override val sort: A
            get() = base.sort

        override fun internEquals(other: Any): Boolean =
            error("Interning is not used for Aux expressions")

        override fun internHashCode(): Int =
            error("Interning is not used for Aux expressions")

        override fun print(printer: ExpressionPrinter) {
            original.print(printer)
        }

        abstract fun getStoreIndex(idx: Int): List<KExpr<*>>

        abstract fun selectValue(originalArray: KExpr<A>, index: List<KExpr<*>>): KExpr<R>
    }

    class SimplifierFlatArrayStoreExpr<D : KSort, R : KSort>(
        ctx: KContext,
        original: KExpr<KArraySort<D, R>>,
        base: KExpr<KArraySort<D, R>>,
        val indices: List<KExpr<D>>,
        values: List<KExpr<R>>,
    ) : SimplifierFlatArrayStoreBaseExpr<KArraySort<D, R>, R>(ctx, indices.size, original, base, values) {
        override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        fun unwrap(): List<KExpr<KSort>> = (listOf(base) + indices + values).uncheckedCast()

        fun wrap(args: List<KExpr<KSort>>) = SimplifierFlatArrayStoreExpr(
            ctx,
            original,
            base = args.first().uncheckedCast(),
            indices = args.subList(
                fromIndex = 1, toIndex = 1 + numIndices
            ).uncheckedCast(),
            values = args.subList(
                fromIndex = 1 + numIndices, toIndex = args.size
            ).uncheckedCast()
        )

        override fun getStoreIndex(idx: Int): List<KExpr<*>> = listOf(indices[idx])

        override fun selectValue(originalArray: KExpr<KArraySort<D, R>>, index: List<KExpr<*>>): KExpr<R> =
            SimplifierFlatArraySelectExpr(
                ctx,
                originalArray,
                base,
                indices,
                values,
                index.single().uncheckedCast()
            )
    }

    class SimplifierFlatArray2StoreExpr<D0 : KSort, D1 : KSort, R : KSort>(
        ctx: KContext,
        original: KExpr<KArray2Sort<D0, D1, R>>,
        base: KExpr<KArray2Sort<D0, D1, R>>,
        val indices0: List<KExpr<D0>>,
        val indices1: List<KExpr<D1>>,
        values: List<KExpr<R>>,
    ) : SimplifierFlatArrayStoreBaseExpr<KArray2Sort<D0, D1, R>, R>(ctx, indices0.size, original, base, values) {
        override fun accept(transformer: KTransformerBase): KExpr<KArray2Sort<D0, D1, R>> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        fun unwrap(): List<KExpr<KSort>> = (listOf(base) + indices0 + indices1 + values).uncheckedCast()

        fun wrap(args: List<KExpr<KSort>>) = SimplifierFlatArray2StoreExpr(
            ctx,
            original,
            base = args.first().uncheckedCast(),
            indices0 = args.subList(
                fromIndex = 1, toIndex = 1 + numIndices
            ).uncheckedCast(),
            indices1 = args.subList(
                fromIndex = 1 + numIndices, toIndex = 1 + 2 * numIndices
            ).uncheckedCast(),
            values = args.subList(
                fromIndex = 1 + 2 * numIndices, toIndex = args.size
            ).uncheckedCast()
        )


        override fun getStoreIndex(idx: Int): List<KExpr<*>> = listOf(indices0[idx], indices1[idx])

        override fun selectValue(originalArray: KExpr<KArray2Sort<D0, D1, R>>, index: List<KExpr<*>>): KExpr<R> =
            SimplifierFlatArray2SelectExpr(
                ctx,
                originalArray,
                base,
                indices0,
                indices1,
                values,
                index.first().uncheckedCast(),
                index.last().uncheckedCast()
            )
    }

    class SimplifierFlatArray3StoreExpr<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort>(
        ctx: KContext,
        original: KExpr<KArray3Sort<D0, D1, D2, R>>,
        base: KExpr<KArray3Sort<D0, D1, D2, R>>,
        val indices0: List<KExpr<D0>>,
        val indices1: List<KExpr<D1>>,
        val indices2: List<KExpr<D2>>,
        values: List<KExpr<R>>,
    ) : SimplifierFlatArrayStoreBaseExpr<KArray3Sort<D0, D1, D2, R>, R>(ctx, indices0.size, original, base, values) {
        override fun accept(transformer: KTransformerBase): KExpr<KArray3Sort<D0, D1, D2, R>> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        fun unwrap(): List<KExpr<KSort>> = (listOf(base) + indices0 + indices1 + indices2 + values).uncheckedCast()

        fun wrap(args: List<KExpr<KSort>>) = SimplifierFlatArray3StoreExpr(
            ctx,
            original,
            base = args.first().uncheckedCast(),
            indices0 = args.subList(
                fromIndex = 1, toIndex = 1 + numIndices
            ).uncheckedCast(),
            indices1 = args.subList(
                fromIndex = 1 + numIndices, toIndex = 1 + 2 * numIndices
            ).uncheckedCast(),
            indices2 = args.subList(
                fromIndex = 1 + 2 * numIndices, toIndex = 1 + 3 * numIndices
            ).uncheckedCast(),
            values = args.subList(
                fromIndex = 1 + 3 * numIndices, toIndex = args.size
            ).uncheckedCast()
        )

        override fun getStoreIndex(idx: Int): List<KExpr<*>> = listOf(indices0[idx], indices1[idx], indices2[idx])

        override fun selectValue(originalArray: KExpr<KArray3Sort<D0, D1, D2, R>>, index: List<KExpr<*>>): KExpr<R> =
            SimplifierFlatArray3SelectExpr(
                ctx,
                originalArray,
                base,
                indices0, indices1, indices2,
                values,
                index[0].uncheckedCast(),
                index[1].uncheckedCast(),
                index[2].uncheckedCast()
            )
    }

    class SimplifierFlatArrayNStoreExpr<R : KSort>(
        ctx: KContext,
        original: KExpr<KArrayNSort<R>>,
        base: KExpr<KArrayNSort<R>>,
        val indices: List<List<KExpr<*>>>,
        values: List<KExpr<R>>,
    ) : SimplifierFlatArrayStoreBaseExpr<KArrayNSort<R>, R>(ctx, indices.size, original, base, values) {
        override fun accept(transformer: KTransformerBase): KExpr<KArrayNSort<R>> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        fun unwrap(): List<KExpr<KSort>> = (listOf(base) + values + indices.flatten()).uncheckedCast()

        fun wrap(args: List<KExpr<KSort>>) = SimplifierFlatArrayNStoreExpr(
            ctx, original,
            base = args.first().uncheckedCast(),
            values = args.subList(
                fromIndex = 1, toIndex = 1 + numIndices
            ).uncheckedCast(),
            indices = args.subList(
                fromIndex = 1 + numIndices, toIndex = args.size
            ).chunked(numIndices)
        )

        override fun getStoreIndex(idx: Int): List<KExpr<*>> = indices[idx]

        override fun selectValue(originalArray: KExpr<KArrayNSort<R>>, index: List<KExpr<*>>): KExpr<R> =
            SimplifierFlatArrayNSelectExpr(
                ctx,
                originalArray,
                base,
                indices,
                values,
                index
            )
    }

    /**
     * Auxiliary expression to handle select with base array expanded.
     * @see [SimplifierAuxExpression]
     * */
    sealed class SimplifierFlatArraySelectBaseExpr<A : KArraySortBase<R>, R : KSort>(
        ctx: KContext,
        val original: KExpr<A>,
        val baseArray: KExpr<A>,
        val storedValues: List<KExpr<R>>
    ) : KExpr<R>(ctx) {
        override val sort: R
            get() = baseArray.sort.range

        override fun internEquals(other: Any): Boolean =
            error("Interning is not used for Aux expressions")

        override fun internHashCode(): Int =
            error("Interning is not used for Aux expressions")

        override fun print(printer: ExpressionPrinter) {
            original.print(printer)
        }

        abstract fun changeBaseArray(newBaseArray: KExpr<A>): SimplifierFlatArraySelectBaseExpr<A, R>
    }

    class SimplifierFlatArraySelectExpr<D : KSort, R : KSort>(
        ctx: KContext,
        original: KExpr<KArraySort<D, R>>,
        baseArray: KExpr<KArraySort<D, R>>,
        val storedIndices: List<KExpr<D>>,
        storedValues: List<KExpr<R>>,
        val index: KExpr<D>,
    ) : SimplifierFlatArraySelectBaseExpr<KArraySort<D, R>, R>(ctx, original, baseArray, storedValues) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        override fun changeBaseArray(newBaseArray: KExpr<KArraySort<D, R>>) =
            SimplifierFlatArraySelectExpr(ctx, original, newBaseArray, storedIndices, storedValues, index)
    }

    class SimplifierFlatArray2SelectExpr<D0 : KSort, D1 : KSort, R : KSort>(
        ctx: KContext,
        original: KExpr<KArray2Sort<D0, D1, R>>,
        baseArray: KExpr<KArray2Sort<D0, D1, R>>,
        val storedIndices0: List<KExpr<D0>>,
        val storedIndices1: List<KExpr<D1>>,
        storedValues: List<KExpr<R>>,
        val index0: KExpr<D0>,
        val index1: KExpr<D1>,
    ) : SimplifierFlatArraySelectBaseExpr<KArray2Sort<D0, D1, R>, R>(ctx, original, baseArray, storedValues) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        override fun changeBaseArray(newBaseArray: KExpr<KArray2Sort<D0, D1, R>>) =
            SimplifierFlatArray2SelectExpr(
                ctx, original, newBaseArray, storedIndices0, storedIndices1, storedValues, index0, index1
            )
    }

    class SimplifierFlatArray3SelectExpr<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort>(
        ctx: KContext,
        original: KExpr<KArray3Sort<D0, D1, D2, R>>,
        baseArray: KExpr<KArray3Sort<D0, D1, D2, R>>,
        val storedIndices0: List<KExpr<D0>>,
        val storedIndices1: List<KExpr<D1>>,
        val storedIndices2: List<KExpr<D2>>,
        storedValues: List<KExpr<R>>,
        val index0: KExpr<D0>,
        val index1: KExpr<D1>,
        val index2: KExpr<D2>,
    ) : SimplifierFlatArraySelectBaseExpr<KArray3Sort<D0, D1, D2, R>, R>(ctx, original, baseArray, storedValues) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        override fun changeBaseArray(newBaseArray: KExpr<KArray3Sort<D0, D1, D2, R>>) =
            SimplifierFlatArray3SelectExpr(
                ctx, original, newBaseArray, storedIndices0, storedIndices1, storedIndices2,
                storedValues, index0, index1, index2
            )
    }

    class SimplifierFlatArrayNSelectExpr<R : KSort>(
        ctx: KContext,
        original: KExpr<KArrayNSort<R>>,
        baseArray: KExpr<KArrayNSort<R>>,
        val storedIndices: List<List<KExpr<*>>>,
        storedValues: List<KExpr<R>>,
        val indices: List<KExpr<*>>,
    ) : SimplifierFlatArraySelectBaseExpr<KArrayNSort<R>, R>(ctx, original, baseArray, storedValues) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }

        override fun changeBaseArray(newBaseArray: KExpr<KArrayNSort<R>>) =
            SimplifierFlatArrayNSelectExpr(ctx, original, newBaseArray, storedIndices, storedValues, indices)
    }

    /**
     * Auxiliary expression to handle array select.
     * @see [SimplifierAuxExpression]
     * */
    sealed class SimplifierArraySelectBaseExpr<A : KArraySortBase<R>, R : KSort>(
        ctx: KContext,
        val array: KExpr<A>
    ) : KExpr<R>(ctx) {
        override val sort: R
            get() = array.sort.range

        override fun internEquals(other: Any): Boolean =
            error("Interning is not used for Aux expressions")

        override fun internHashCode(): Int =
            error("Interning is not used for Aux expressions")

        override fun print(printer: ExpressionPrinter) {
            printer.append("(simplifierSelect ")
            printer.append(array)
            printer.append(")")
        }
    }

    class SimplifierArraySelectExpr<D : KSort, R : KSort>(
        ctx: KContext,
        array: KExpr<KArraySort<D, R>>,
        val index: KExpr<D>,
    ) : SimplifierArraySelectBaseExpr<KArraySort<D, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    class SimplifierArray2SelectExpr<D0 : KSort, D1 : KSort, R : KSort>(
        ctx: KContext,
        array: KExpr<KArray2Sort<D0, D1, R>>,
        val index0: KExpr<D0>,
        val index1: KExpr<D1>
    ) : SimplifierArraySelectBaseExpr<KArray2Sort<D0, D1, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    class SimplifierArray3SelectExpr<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort>(
        ctx: KContext,
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        val index0: KExpr<D0>,
        val index1: KExpr<D1>,
        val index2: KExpr<D2>,
    ) : SimplifierArraySelectBaseExpr<KArray3Sort<D0, D1, D2, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    class SimplifierArrayNSelectExpr<R : KSort>(
        ctx: KContext,
        array: KExpr<KArrayNSort<R>>,
        val indices: List<KExpr<*>>,
    ) : SimplifierArraySelectBaseExpr<KArrayNSort<R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }
}
