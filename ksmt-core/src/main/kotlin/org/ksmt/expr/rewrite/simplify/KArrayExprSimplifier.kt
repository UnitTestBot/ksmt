package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArrayLambda
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KEqExpr
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

interface KArrayExprSimplifier : KExprSimplifierBase {

    fun <D : KSort, R : KSort> simplifyEqArray(
        lhs: KExpr<KArraySort<D, R>>,
        rhs: KExpr<KArraySort<D, R>>
    ): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        val leftArray = flatStores(lhs)
        val rightArray = flatStores(rhs)
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
        if (lBase == rBase || lBase is KArrayConst<D, R> && rBase is KArrayConst<D, R>) {
            val simplifiedExpr = simplifyArrayStoreEq(lhs, leftArray, rhs, rightArray)
            return rewrite(simplifiedExpr)
        }

        return mkEq(lhs, rhs)
    }

    /**
     * (= (store a i v) (store b x y)) ==>
     * (and
     *   (= (select (store a i v) i) (select (store b x y) i))
     *   (= (select (store a i v) x) (select (store b x y) x))
     *   (= a b)
     * )
     */
    private fun <D : KSort, R : KSort> simplifyArrayStoreEq(
        lhs: KExpr<KArraySort<D, R>>,
        leftArray: SimplifierFlatArrayStoreExpr<D, R>,
        rhs: KExpr<KArraySort<D, R>>,
        rightArray: SimplifierFlatArrayStoreExpr<D, R>,
    ): SimplifierAuxExpression<KBoolSort> = auxExpr {
        val checks = arrayListOf<KExpr<KBoolSort>>()
        if (leftArray.base is KArrayConst<D, R> && rightArray.base is KArrayConst<D, R>) {
            // (= (const a) (const b)) ==> (= a b)
            checks += KEqExpr(ctx, leftArray.base.value, rightArray.base.value)
        } else {
            check(leftArray.base == rightArray.base) {
                "Base arrays expected to be equal or constant"
            }
        }

        val leftArraySearchInfo = analyzeArrayStores(leftArray)
        val rightArraySearchInfo = analyzeArrayStores(rightArray)

        val allIndices = leftArray.indices.toSet() + rightArray.indices.toSet()
        for (idx in allIndices) {
            val lValue = selectArrayValue(idx, leftArraySearchInfo, lhs, leftArray)
            val rValue = selectArrayValue(idx, rightArraySearchInfo, rhs, rightArray)
            if (lValue != rValue) {
                checks += KEqExpr(ctx, lValue, rValue)
            }
        }

        KAndExpr(ctx, checks)
    }

    private class ArrayStoreSearchInfo<D : KSort, R : KSort>(
        val storeIndices: Map<KExpr<D>, Int>,
        val storeValues: List<KExpr<R>>,
        val nonConstants: List<KExpr<D>>,
        val nonConstantsToCheck: IntArray
    )

    private fun <D : KSort, R : KSort> selectArrayValue(
        selectIndex: KExpr<D>,
        arraySearchInfo: ArrayStoreSearchInfo<D, R>,
        originalArrayExpr: KExpr<KArraySort<D, R>>,
        flatArray: SimplifierFlatArrayStoreExpr<D, R>
    ): KExpr<R> =
        findStoredArrayValue(arraySearchInfo, selectIndex)
            ?: SimplifierFlatArraySelectExpr(
                ctx,
                originalArrayExpr,
                flatArray.base,
                flatArray.indices,
                flatArray.values,
                selectIndex
            )

    /**
     * Preprocess flat array stores for faster index search.
     * @see [findStoredArrayValue]
     * */
    private fun <D : KSort, R : KSort> analyzeArrayStores(
        array: SimplifierFlatArrayStoreExpr<D, R>
    ): ArrayStoreSearchInfo<D, R> {
        val indices = array.indices
        val indexId = hashMapOf<KExpr<D>, Int>()
        val nonConstants = arrayListOf<KExpr<D>>()
        val nonConstantsToCheck = IntArray(indices.size)

        for (i in indices.indices) {
            val storeIndex = indices[i]
            if (storeIndex !in indexId) {
                indexId[storeIndex] = i
                if (!storeIndex.definitelyIsConstant) {
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
    private fun <D : KSort, R : KSort> findStoredArrayValue(
        array: ArrayStoreSearchInfo<D, R>,
        selectIndex: KExpr<D>,
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

    @Suppress("UNCHECKED_CAST")
    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        simplifyApp(
            expr = expr as KApp<KArraySort<D, R>, KExpr<KSort>>,
            preprocess = { flatStores(expr) }
        ) {
            error("Always preprocessed")
        }

    @Suppress("UNCHECKED_CAST")
    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        simplifyApp(
            expr = expr as KApp<R, KExpr<KSort>>,
            preprocess = {
                val array = flatStores(expr.array)
                SimplifierFlatArraySelectExpr(
                    ctx,
                    original = expr.array,
                    baseArray = array.base,
                    storedValues = array.values,
                    storedIndices = array.indices,
                    index = expr.index
                )
            }
        ) {
            error("Always preprocessed")
        }

    @Suppress("ComplexMethod", "LoopWithTooManyJumpStatements")
    private fun <D : KSort, R : KSort> transform(expr: SimplifierFlatArrayStoreExpr<D, R>): KExpr<KArraySort<D, R>> =
        simplifyApp(expr) { transformedArgs ->
            val base: KExpr<KArraySort<D, R>> = transformedArgs.first().uncheckedCast()
            val indices: List<KExpr<D>> = transformedArgs.subList(
                fromIndex = 1, toIndex = 1 + expr.indices.size
            ).uncheckedCast()
            val values: List<KExpr<R>> = transformedArgs.subList(
                fromIndex = 1 + expr.indices.size, toIndex = transformedArgs.size
            ).uncheckedCast()

            if (indices.isEmpty()) {
                return@simplifyApp base
            }

            val storedIndices = hashMapOf<KExpr<D>, Int>()
            val simplifiedIndices = arrayListOf<KExpr<D>>()
            val simplifiedValues = arrayListOf<KExpr<R>>()
            var lastNonConstantIndex = -1

            for (i in indices.indices.reversed()) {
                val index = indices[i]
                val value = values[i]

                val storedIndexPosition = storedIndices[index]
                if (storedIndexPosition != null) {

                    /** Try push store.
                     * (store (store a i x) j y), i != j ==> (store (store a j y) i x)
                     *
                     * Store can be pushed only if all parent indices are definitely not equal.
                     * */
                    val allParentIndicesAreDistinct = allParentSoreIndicesAreDistinct(
                        index = index,
                        storedIndexPosition = storedIndexPosition,
                        lastNonConstantIndex = lastNonConstantIndex,
                        simplifiedIndices = simplifiedIndices
                    )

                    if (allParentIndicesAreDistinct) {
                        simplifiedValues[storedIndexPosition] = value
                        if (!index.definitelyIsConstant) {
                            lastNonConstantIndex = maxOf(lastNonConstantIndex, storedIndexPosition)
                        }
                        continue
                    }
                }

                // simplify direct store to array
                if (storedIndices.isEmpty()) {
                    // (store (const v) i v) ==> (const v)
                    if (base is KArrayConst<D, R> && base.value == value) {
                        continue
                    }

                    // (store a i (select a i)) ==> a
                    if (value is KArraySelect<*, *> && base == value.array && index == value.index) {
                        continue
                    }
                }

                // store
                simplifiedIndices.add(index)
                simplifiedValues.add(value)
                val indexPosition = simplifiedIndices.lastIndex
                storedIndices[index] = indexPosition
                if (!index.definitelyIsConstant) {
                    lastNonConstantIndex = maxOf(lastNonConstantIndex, indexPosition)
                }
            }

            var result = base
            for (i in simplifiedIndices.indices) {
                val index = simplifiedIndices[i]
                val value = simplifiedValues[i]
                result = result.store(index, value)
            }

            return@simplifyApp result
        }

    private fun <D : KSort> allParentSoreIndicesAreDistinct(
        index: KExpr<D>,
        storedIndexPosition: Int,
        lastNonConstantIndex: Int,
        simplifiedIndices: List<KExpr<D>>
    ): Boolean {
        /**
         * Since all constants are trivially comparable we can guarantee, that
         * all parents are distinct.
         * Otherwise, we need to perform full check of parent indices.
         * */
        if (index.definitelyIsConstant && storedIndexPosition >= lastNonConstantIndex) return true

        /**
         *  If non-constant index is among the indices we need to check,
         *  we can check it first. Since non-constant indices are usually
         *  not definitely distinct, we will not check all other indices.
         * */
        if (lastNonConstantIndex > storedIndexPosition
            && !areDefinitelyDistinct(index, simplifiedIndices[lastNonConstantIndex])
        ) {
            return false
        }

        for (checkIdx in (storedIndexPosition + 1) until simplifiedIndices.size) {
            if (!areDefinitelyDistinct(index, simplifiedIndices[checkIdx])) {
                // possibly equal index, we can't squash stores
                return false
            }
        }

        return true
    }

    /**
     * Try to simplify only indices first.
     * Usually this will be enough and we will not produce many irrelevant array store expressions.
     * */
    private fun <D : KSort, R : KSort> transform(expr: SimplifierFlatArraySelectExpr<D, R>): KExpr<R> =
        simplifyApp(expr) { allIndices ->
            val index = allIndices.first()
            val arrayIndices = allIndices.subList(fromIndex = 1, toIndex = allIndices.size)

            var arrayStoreIdx = 0
            while (arrayStoreIdx < arrayIndices.size) {
                val storeIdx = arrayIndices[arrayStoreIdx]

                // (select (store i v) i) ==> v
                if (storeIdx == index) {
                    return@simplifyApp rewrite(expr.storedValues[arrayStoreIdx])
                }

                if (!areDefinitelyDistinct(index, storeIdx)) {
                    break
                }

                // (select (store a i v) j), i != j ==> (select a j)
                arrayStoreIdx++
            }

            if (arrayStoreIdx == arrayIndices.size) {
                return@simplifyApp rewrite(SimplifierArraySelectExpr(ctx, expr.baseArray, index))
            }

            rewrite(SimplifierArraySelectExpr(ctx, expr.original, index))
        }

    private fun <D : KSort, R : KSort> transform(expr: SimplifierArraySelectExpr<D, R>): KExpr<R> =
        simplifyApp(expr) { (arrayArg, indexArg) ->
            var array: KExpr<KArraySort<D, R>> = arrayArg.uncheckedCast()
            val index: KExpr<D> = indexArg.uncheckedCast()

            while (array is KArrayStore<D, R>) {
                // (select (store i v) i) ==> v
                if (array.index == index) {
                    return@simplifyApp array.value
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
                    mkArraySelect(array, index)
                }
            }
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
    private class SimplifierFlatArrayStoreExpr<D : KSort, R : KSort>(
        ctx: KContext,
        val base: KExpr<KArraySort<D, R>>,
        val indices: List<KExpr<D>>,
        val values: List<KExpr<R>>,
    ) : KApp<KArraySort<D, R>, KExpr<KSort>>(ctx) {

        override val args: List<KExpr<KSort>> =
            (listOf(base) + indices + values).uncheckedCast()

        override val decl: KDecl<KArraySort<D, R>>
            get() = ctx.mkArrayStoreDecl(base.sort)

        override val sort: KArraySort<D, R>
            get() = base.sort

        override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to handle select with base array expanded.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierFlatArraySelectExpr<D : KSort, R : KSort>(
        ctx: KContext,
        val original: KExpr<KArraySort<D, R>>,
        val baseArray: KExpr<KArraySort<D, R>>,
        val storedIndices: List<KExpr<D>>,
        val storedValues: List<KExpr<R>>,
        val index: KExpr<D>,
    ) : KApp<R, KExpr<D>>(ctx) {

        override val args: List<KExpr<D>> =
            listOf(index) + storedIndices

        override val sort: R
            get() = baseArray.sort.range

        override val decl: KDecl<R>
            get() = ctx.mkArraySelectDecl(baseArray.sort)

        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

    /**
     * Auxiliary expression to handle array select.
     * @see [SimplifierAuxExpression]
     * */
    private class SimplifierArraySelectExpr<D : KSort, R : KSort>(
        ctx: KContext,
        val array: KExpr<KArraySort<D, R>>,
        val index: KExpr<D>,
    ) : KApp<R, KExpr<KSort>>(ctx) {
        override val args: List<KExpr<KSort>> = listOf(array, index).uncheckedCast()

        override val sort: R
            get() = array.sort.range

        override val decl: KDecl<R>
            get() = ctx.mkArraySelectDecl(array.sort)

        override fun accept(transformer: KTransformerBase): KExpr<R> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

}
