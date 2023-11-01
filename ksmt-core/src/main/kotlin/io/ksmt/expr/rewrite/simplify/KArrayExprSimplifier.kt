package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KArray2Select
import io.ksmt.expr.KArray2Store
import io.ksmt.expr.KArray3Select
import io.ksmt.expr.KArray3Store
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayNSelect
import io.ksmt.expr.KArrayNStore
import io.ksmt.expr.KArraySelect
import io.ksmt.expr.KArraySelectBase
import io.ksmt.expr.KArrayStore
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KEqExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

@Suppress("LargeClass")
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

        return withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
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

        ctx.mkAndAuxExpr(checks)
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
            if (!areDefinitelyDistinct(selectIndex, nonConstant)) {
                return null
            }
        }

        return array.storeValues[storeIndex]
    }

    fun <D : KSort, R : KSort> KContext.preprocess(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> = flatStores(expr)

    fun <D : KSort, R : KSort> KContext.postRewriteArrayStore(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
        value: KExpr<R>
    ): KExpr<KArraySort<D, R>> = mkArrayStoreNoSimplify(array, index, value)

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        simplifyExpr(
            expr = expr,
            a0 = expr.array,
            a1 = expr.index,
            a2 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { array, index, value -> postRewriteArrayStore(array, index, value) }
        )

    fun <D0 : KSort, D1 : KSort, R : KSort> KContext.preprocess(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> = flatStores2(expr)

    fun <D0 : KSort, D1 : KSort, R : KSort> KContext.postRewriteArrayStore(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        value: KExpr<R>
    ): KExpr<KArray2Sort<D0, D1, R>> = mkArrayStoreNoSimplify(array, index0, index1, value)

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: KArray2Store<D0, D1, R>
    ): KExpr<KArray2Sort<D0, D1, R>> =
        simplifyExpr(
            expr = expr,
            a0 = expr.array,
            a1 = expr.index0,
            a2 = expr.index1,
            a3 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { array, index0, index1, value -> postRewriteArrayStore(array, index0, index1, value) }
        )

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.preprocess(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = flatStores3(expr)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.postRewriteArrayStore(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
        index2: KExpr<D2>,
        value: KExpr<R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> = mkArrayStoreNoSimplify(array, index0, index1, index2, value)

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Store<D0, D1, D2, R>
    ): KExpr<KArray3Sort<D0, D1, D2, R>> =
        simplifyExpr(
            expr = expr,
            a0 = expr.array,
            a1 = expr.index0,
            a2 = expr.index1,
            a3 = expr.index2,
            a4 = expr.value,
            preprocess = { preprocess(it) },
            simplifier = { array, i0, i1, i2, value -> postRewriteArrayStore(array, i0, i1, i2, value) }
        )

    fun <R : KSort> KContext.preprocess(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> = flatStoresN(expr)

    fun <R : KSort> KContext.postRewriteArrayNStore(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<KSort>>,
        value: KExpr<R>
    ): KExpr<KArrayNSort<R>> = mkArrayNStoreNoSimplify(array, indices, value)

    override fun <R : KSort> transform(expr: KArrayNStore<R>): KExpr<KArrayNSort<R>> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = {
                postRewriteArrayNStore(
                    it.first().uncheckedCast(),
                    it.subList(fromIndex = 1, toIndex = it.size - 1),
                    it.last()
                ).uncheckedCast()
            }
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
                ctx.postRewriteArrayStore(array, simplifiedIndices[simplifiedIdx], value)
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
                    || areDefinitelyDistinct(indices1[i], simplifiedIndices1[simplifiedIdx])
            },
            selectIndicesMatch = { select: KArray2Select<D0, D1, R>, i ->
                indices0[i] == select.index0 && indices1[i] == select.index1
            },
            mkSimplifiedStore = { array, simplifiedIdx, value ->
                ctx.postRewriteArrayStore(
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
                    || areDefinitelyDistinct(indices1[i], simplifiedIndices1[simplifiedIdx])
                    || areDefinitelyDistinct(indices2[i], simplifiedIndices2[simplifiedIdx])
            },
            selectIndicesMatch = { select: KArray3Select<D0, D1, D2, R>, i ->
                indices0[i] == select.index0
                    && indices1[i] == select.index1
                    && indices2[i] == select.index2
            },
            mkSimplifiedStore = { array, simplifiedIdx, value ->
                ctx.postRewriteArrayStore(
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
                ctx.postRewriteArrayNStore(array, simplifiedIndices[simplifiedIdx].uncheckedCast(), value)
            }
        )
    }

    @Suppress("LongParameterList", "NestedBlockDepth", "ComplexMethod", "LoopWithTooManyJumpStatements")
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

    @Suppress("LongParameterList")
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

    fun <D : KSort, R : KSort> KContext.preprocess(expr: KArraySelect<D, R>): KExpr<R> = expr

    fun <D : KSort, R : KSort> KContext.postRewriteArraySelect(
        array: KExpr<KArraySort<D, R>>,
        index: KExpr<D>,
    ): KExpr<R> = simplifySelectFromArrayStore(
        array = array,
        index = index,
        storeIndexMatch = { store: KArrayStore<D, R>, idx: KExpr<D> -> idx == store.index },
        storeIndexDistinct = { store: KArrayStore<D, R>, idx: KExpr<D> ->
            areDefinitelyDistinct(idx, store.index)
        }
    ) { array2, i ->
        simplifyArraySelectLambda(array2, i, { rewrite(it) }, ::mkArraySelectNoSimplify)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        simplifyExpr(
            expr = expr,
            a0 = expr.index,
            preprocess = { preprocess(it) }
        ) { index ->
            transformSelect(expr.array, index)
        }

    fun <D0 : KSort, D1 : KSort, R : KSort> KContext.preprocess(expr: KArray2Select<D0, D1, R>): KExpr<R> = expr

    fun <D0 : KSort, D1 : KSort, R : KSort> KContext.postRewriteArraySelect(
        array: KExpr<KArray2Sort<D0, D1, R>>,
        index0: KExpr<D0>,
        index1: KExpr<D1>,
    ): KExpr<R> = simplifySelectFromArrayStore(
        array = array,
        index0 = index0,
        index1 = index1,
        storeIndexMatch = { store: KArray2Store<D0, D1, R>, idx0: KExpr<D0>, idx1: KExpr<D1> ->
            idx0 == store.index0 && idx1 == store.index1
        },
        storeIndexDistinct = { store: KArray2Store<D0, D1, R>, idx0: KExpr<D0>, idx1: KExpr<D1> ->
            areDefinitelyDistinct(idx0, store.index0)
                    || areDefinitelyDistinct(idx1, store.index1)
        }
    ) { array2, i0, i1 ->
        simplifyArraySelectLambda(array2, i0, i1, { rewrite(it) }, ::mkArraySelectNoSimplify)
    }

    override fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: KArray2Select<D0, D1, R>): KExpr<R> =
        simplifyExpr(
            expr = expr,
            a0 = expr.index0,
            a1 = expr.index1,
            preprocess = { preprocess(it) }
        ) { index0, index1 ->
            transformSelect(expr.array, index0, index1)
        }

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.preprocess(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = expr

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> KContext.postRewriteArraySelect(
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
            areDefinitelyDistinct(idx0, store.index0)
                    || areDefinitelyDistinct(idx1, store.index1)
                    || areDefinitelyDistinct(idx2, store.index2)
        }
    ) { array2, i0, i1, i2 ->
        simplifyArraySelectLambda(array2, i0, i1, i2, { rewrite(it) }, ::mkArraySelectNoSimplify)
    }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: KArray3Select<D0, D1, D2, R>
    ): KExpr<R> = simplifyExpr(
        expr = expr,
        a0 = expr.index0,
        a1 = expr.index1,
        a2 = expr.index2,
        preprocess = { preprocess(it) }
    ) { index0, index1, index2 ->
        transformSelect(expr.array, index0, index1, index2)
    }

    fun <R : KSort> KContext.preprocess(expr: KArrayNSelect<R>): KExpr<R> = expr

    fun <R : KSort> KContext.postRewriteArrayNSelect(
        array: KExpr<KArrayNSort<R>>,
        indices: List<KExpr<KSort>>,
    ): KExpr<R> = simplifyArrayNSelectFromArrayStore(
        array = array,
        indices = indices,
        storeIndexMatch = { store: KArrayNStore<R>, idxs: List<KExpr<*>> -> store.indices == idxs },
        storeIndexDistinct = { store: KArrayNStore<R>, idxs: List<KExpr<*>> ->
            areDefinitelyDistinct(idxs, store.indices)
        }
    ) { array2, indices2 ->
        simplifyArrayNSelectLambda(array2, indices2, { rewrite(it) }, ::mkArrayNSelectNoSimplify)
    }

    override fun <R : KSort> transform(expr: KArrayNSelect<R>): KExpr<R> =
        simplifyExpr(
            expr = expr,
            args = expr.indices,
            preprocess = { preprocess(it) }
        ) { indices ->
            transformSelect(expr.array, indices)
        }

    fun <D : KSort, R : KSort> transformSelect(
        array: KExpr<KArraySort<D, R>>, index: KExpr<D>
    ): KExpr<R> = transformSelect(
        array,
        findArrayToSelectFrom = { store: KArrayStore<D, R> -> store.findArrayToSelectFrom(index) },
        selectFromStore = { store: KArrayStore<D, R> -> SelectFromStoreExpr(ctx, store, index) },
        selectFromFunction = { f -> KFunctionApp(ctx, f, listOf(index).uncheckedCast()) },
        default = { a -> SimplifierArraySelectExpr(ctx, a, index) }
    )

    fun <D0 : KSort, D1 : KSort, R : KSort> transformSelect(
        array: KExpr<KArray2Sort<D0, D1, R>>, index0: KExpr<D0>, index1: KExpr<D1>
    ): KExpr<R> = transformSelect(
        array,
        findArrayToSelectFrom = { store: KArray2Store<D0, D1, R> -> store.findArrayToSelectFrom(index0, index1) },
        selectFromStore = { store: KArray2Store<D0, D1, R> -> Select2FromStoreExpr(ctx, store, index0, index1) },
        selectFromFunction = { f -> KFunctionApp(ctx, f, listOf(index0, index1).uncheckedCast()) },
        default = { a -> SimplifierArray2SelectExpr(ctx, a, index0, index1) }
    )

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transformSelect(
        array: KExpr<KArray3Sort<D0, D1, D2, R>>,
        index0: KExpr<D0>, index1: KExpr<D1>, index2: KExpr<D2>
    ): KExpr<R> = transformSelect(
        array,
        findArrayToSelectFrom = { store: KArray3Store<D0, D1, D2, R> ->
            store.findArrayToSelectFrom(index0, index1, index2)
        },
        selectFromStore = { store: KArray3Store<D0, D1, D2, R> ->
            Select3FromStoreExpr(ctx, store, index0, index1, index2)
        },
        selectFromFunction = { f -> KFunctionApp(ctx, f, listOf(index0, index1, index2).uncheckedCast()) },
        default = { a -> SimplifierArray3SelectExpr(ctx, a, index0, index1, index2) }
    )

    fun <R : KSort> transformSelect(
        array: KExpr<KArrayNSort<R>>, indices: List<KExpr<KSort>>
    ): KExpr<R> = transformSelect(
        array,
        findArrayToSelectFrom = { store: KArrayNStore<R> -> store.findArrayToSelectFrom(indices) },
        selectFromStore = { store: KArrayNStore<R> -> SelectNFromStoreExpr(ctx, store, indices) },
        selectFromFunction = { f -> KFunctionApp(ctx, f, indices) },
        default = { a -> SimplifierArrayNSelectExpr(ctx, a, indices) }
    )

    private inline fun <A : KArraySortBase<R>, R : KSort, reified S : KArrayStoreBase<A, R>> transformSelect(
        array: KExpr<A>,
        findArrayToSelectFrom: (S) -> KExpr<A>,
        selectFromStore: (S) -> KExpr<R>,
        selectFromFunction: (KDecl<R>) -> KExpr<R>,
        default: (KExpr<A>) -> KExpr<R>
    ): KExpr<R> = when (val arrayToSelect = if (array is S) findArrayToSelectFrom(array) else array) {
        is S -> rewrite(selectFromStore(arrayToSelect))
        is KFunctionAsArray<A, *> -> rewrite(selectFromFunction(arrayToSelect.function.uncheckedCast()))
        else -> rewrite(default(arrayToSelect))
    }

    private fun <D : KSort, R : KSort> transform(expr: SelectFromStoreExpr<D, R>): KExpr<R> =
        simplifyExpr(expr, expr.array.index) { storeIndex ->
            transformSelectFromStore(
                expr = expr,
                indexMatch = { expr.index == storeIndex },
                indexDistinct = { areDefinitelyDistinct(expr.index, storeIndex) },
                transformNested = { transformSelect(expr.array.array, expr.index) },
                default = { SimplifierArraySelectExpr(ctx, expr.array, expr.index) }
            )
        }

    private fun <D0 : KSort, D1 : KSort, R : KSort> transform(expr: Select2FromStoreExpr<D0, D1, R>): KExpr<R> =
        simplifyExpr(expr, expr.array.index0, expr.array.index1) { storeIndex0, storeIndex1 ->
            transformSelectFromStore(
                expr = expr,
                indexMatch = { expr.index0 == storeIndex0 && expr.index1 == storeIndex1 },
                indexDistinct = {
                    areDefinitelyDistinct(expr.index0, storeIndex0)
                        || areDefinitelyDistinct(expr.index1, storeIndex1)
                },
                transformNested = { transformSelect(expr.array.array, expr.index0, expr.index1) },
                default = { SimplifierArray2SelectExpr(ctx, expr.array, expr.index0, expr.index1) }
            )
        }

    private fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: Select3FromStoreExpr<D0, D1, D2, R>
    ): KExpr<R> =
        simplifyExpr(expr, expr.array.index0, expr.array.index1, expr.array.index2) { si0, si1, si2 ->
            transformSelectFromStore(
                expr = expr,
                indexMatch = { expr.index0 == si0 && expr.index1 == si1 && expr.index2 == si2 },
                indexDistinct = {
                    areDefinitelyDistinct(expr.index0, si0)
                        || areDefinitelyDistinct(expr.index1, si1)
                        || areDefinitelyDistinct(expr.index2, si2)
                },
                transformNested = { transformSelect(expr.array.array, expr.index0, expr.index1, expr.index2) },
                default = { SimplifierArray3SelectExpr(ctx, expr.array, expr.index0, expr.index1, expr.index2) }
            )
        }

    private fun <R : KSort> transform(expr: SelectNFromStoreExpr<R>): KExpr<R> =
        simplifyExpr(expr, expr.array.indices) { indices ->
            transformSelectFromStore(
                expr = expr,
                indexMatch = { expr.indices == indices },
                indexDistinct = { areDefinitelyDistinct(expr.indices, indices) },
                transformNested = { transformSelect(expr.array.array, expr.indices) },
                default = { SimplifierArrayNSelectExpr(ctx, expr.array, expr.indices) }
            )
        }

    private inline fun <A : KArraySortBase<R>, R : KSort> transformSelectFromStore(
        expr: SelectFromStoreExprBase<A, R>,
        indexMatch: () -> Boolean,
        indexDistinct: () -> Boolean,
        transformNested: () -> KExpr<R>,
        default: () -> KExpr<R>
    ): KExpr<R> {
        // (select (store i v) i) ==> v
        if (indexMatch()) {
            return rewrite(expr.storeValue)
        }

        // (select (store a i v) j), i != j ==> (select a j)
        if (indexDistinct()) {
            return transformNested()
        }

        return rewrite(default())
    }

    private fun <D : KSort, R : KSort> transform(expr: SimplifierArraySelectExpr<D, R>): KExpr<R> =
        simplifyExpr(expr, expr.array) { array ->
            postRewriteArraySelect(array, expr.index)
        }

    private fun <D0 : KSort, D1 : KSort, R : KSort> transform(
        expr: SimplifierArray2SelectExpr<D0, D1, R>
    ): KExpr<R> = simplifyExpr(expr, expr.array) { array ->
        postRewriteArraySelect(array, expr.index0, expr.index1)
    }

    private fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> transform(
        expr: SimplifierArray3SelectExpr<D0, D1, D2, R>
    ): KExpr<R> = simplifyExpr(expr, expr.array) { array ->
        postRewriteArraySelect(array, expr.index0, expr.index1, expr.index2)
    }

    private fun <R : KSort> transform(expr: SimplifierArrayNSelectExpr<R>): KExpr<R> =
        simplifyExpr(expr, expr.array) { array ->
            postRewriteArrayNSelect(array, expr.indices.uncheckedCast())
        }

    @Suppress("USELESS_CAST") // Exhaustive when
    private fun <A : KArraySortBase<R>, R : KSort> flatStoresGeneric(
        sort: A, expr: KExpr<A>
    ): SimplifierFlatArrayStoreBaseExpr<A, R> = when (sort as KArraySortBase<R>) {
        is KArraySort<*, R> -> flatStores<KSort, R>(expr.uncheckedCast()).uncheckedCast()
        is KArray2Sort<*, *, R> -> flatStores2<KSort, KSort, R>(expr.uncheckedCast()).uncheckedCast()
        is KArray3Sort<*, *, *, R> -> flatStores3<KSort, KSort, KSort, R>(expr.uncheckedCast()).uncheckedCast()
        is KArrayNSort<R> -> flatStoresN<R>(expr.uncheckedCast()).uncheckedCast()
    }

    private fun <D : KSort, R : KSort> flatStores(
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

    /**
     * Auxiliary expression to handle expanded array stores.
     * @see [SimplifierAuxExpression]
     * */
    private sealed class SimplifierFlatArrayStoreBaseExpr<A : KArraySortBase<R>, R : KSort>(
        ctx: KContext,
        val numIndices: Int,
        val original: KExpr<A>,
        val base: KExpr<A>,
        val values: List<KExpr<R>>
    ) : KExpr<A>(ctx), KSimplifierAuxExpr {
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

    private class SimplifierFlatArrayStoreExpr<D : KSort, R : KSort>(
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
            KArraySelect(ctx, originalArray, index.single().uncheckedCast())
    }

    private class SimplifierFlatArray2StoreExpr<D0 : KSort, D1 : KSort, R : KSort>(
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

        fun unwrap(): List<KExpr<KSort>> = buildList {
            add(base)
            addAll(indices0)
            addAll(indices1)
            addAll(values)
        }.uncheckedCast()

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
            KArray2Select(ctx, originalArray, index.first().uncheckedCast(), index.last().uncheckedCast())
    }

    @Suppress("LongParameterList")
    private class SimplifierFlatArray3StoreExpr<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort>(
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

        fun unwrap(): List<KExpr<KSort>> = buildList {
            add(base)
            addAll(indices0)
            addAll(indices1)
            addAll(indices2)
            addAll(values)
        }.uncheckedCast()

        @Suppress("MagicNumber")
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

        override fun selectValue(originalArray: KExpr<KArray3Sort<D0, D1, D2, R>>, index: List<KExpr<*>>): KExpr<R> {
            val (idx0, idx1, idx2) = index
            return KArray3Select(
                ctx,
                originalArray,
                idx0.uncheckedCast(),
                idx1.uncheckedCast(),
                idx2.uncheckedCast()
            )
        }
    }

    private class SimplifierFlatArrayNStoreExpr<R : KSort>(
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

        fun unwrap(): List<KExpr<KSort>> = buildList {
            add(base)
            addAll(values)
            this@SimplifierFlatArrayNStoreExpr.indices.forEach {
                addAll(it)
            }
        }.uncheckedCast()

        fun wrap(args: List<KExpr<KSort>>) = SimplifierFlatArrayNStoreExpr(
            ctx, original,
            base = args.first().uncheckedCast(),
            values = args.subList(
                fromIndex = 1, toIndex = 1 + numIndices
            ).uncheckedCast(),
            indices = args.subList(
                fromIndex = 1 + numIndices, toIndex = args.size
            ).chunked(original.sort.domainSorts.size)
        )

        override fun getStoreIndex(idx: Int): List<KExpr<*>> = indices[idx]

        override fun selectValue(originalArray: KExpr<KArrayNSort<R>>, index: List<KExpr<*>>): KExpr<R> =
            KArrayNSelect(ctx, originalArray, index.uncheckedCast())
    }

    /**
     * Auxiliary expression to handle array select.
     * @see [SimplifierAuxExpression]
     * */
    private sealed class SimplifierArraySelectBaseExpr<A : KArraySortBase<R>, R : KSort>(
        ctx: KContext,
        val array: KExpr<A>
    ) : KExpr<R>(ctx), KSimplifierAuxExpr {
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

    private class SimplifierArraySelectExpr<D : KSort, R : KSort>(
        ctx: KContext,
        array: KExpr<KArraySort<D, R>>,
        val index: KExpr<D>,
    ) : SimplifierArraySelectBaseExpr<KArraySort<D, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    private class SimplifierArray2SelectExpr<D0 : KSort, D1 : KSort, R : KSort>(
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

    private class SimplifierArray3SelectExpr<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort>(
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

    private class SimplifierArrayNSelectExpr<R : KSort>(
        ctx: KContext,
        array: KExpr<KArrayNSort<R>>,
        val indices: List<KExpr<*>>,
    ) : SimplifierArraySelectBaseExpr<KArrayNSort<R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to handle select from array store.
     * @see [SimplifierAuxExpression]
     * */
    private sealed class SelectFromStoreExprBase<A : KArraySortBase<R>, R : KSort>(
        ctx: KContext,
        array: KArrayStoreBase<A, R>
    ) : KExpr<R>(ctx), KSimplifierAuxExpr {
        val storeValue: KExpr<R> = array.value

        override val sort: R = storeValue.sort

        override fun print(printer: ExpressionPrinter) {
            printer.append("(selectFromStore)")
        }

        override fun internEquals(other: Any): Boolean =
            error("Interning is not used for Aux expressions")

        override fun internHashCode(): Int =
            error("Interning is not used for Aux expressions")
    }

    private class SelectFromStoreExpr<D : KSort, R : KSort>(
        ctx: KContext,
        val array: KArrayStore<D, R>,
        val index: KExpr<D>,
    ) : SelectFromStoreExprBase<KArraySort<D, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    private class Select2FromStoreExpr<D0 : KSort, D1 : KSort, R : KSort>(
        ctx: KContext,
        val array: KArray2Store<D0, D1, R>,
        val index0: KExpr<D0>,
        val index1: KExpr<D1>
    ) : SelectFromStoreExprBase<KArray2Sort<D0, D1, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    private class Select3FromStoreExpr<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort>(
        ctx: KContext,
        val array: KArray3Store<D0, D1, D2, R>,
        val index0: KExpr<D0>,
        val index1: KExpr<D1>,
        val index2: KExpr<D2>,
    ) : SelectFromStoreExprBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    private class SelectNFromStoreExpr<R : KSort>(
        ctx: KContext,
        val array: KArrayNStore<R>,
        val indices: List<KExpr<KSort>>,
    ) : SelectFromStoreExprBase<KArrayNSort<R>, R>(ctx, array) {
        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }
}
