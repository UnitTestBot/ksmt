package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFalse
import org.ksmt.expr.KFpValue
import org.ksmt.expr.KIntNumExpr
import org.ksmt.expr.KRealNumExpr
import org.ksmt.expr.KTrue
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

interface KArrayExprSimplifier : KExprSimplifierBase {

    fun <T : KArraySort<*, *>> simplifyEqArray(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        lhs as KExpr<KArraySort<*, *>>
        rhs as KExpr<KArraySort<*, *>>

        val (lStores, lBase) = expandStores(lhs)
        val (rStores, rBase) = expandStores(rhs)

        /**
         * (= (store a i v) (store b x y)) ==>
         * (and
         *   (= (select a i) (select b i))
         *   (= (select a x) (select b x))
         *   (= a b)
         * )
         */
        if (lBase == rBase || lBase is KArrayConst<*, *> && rBase is KArrayConst<*, *>) {
            val checks = arrayListOf<KExpr<KBoolSort>>()
            if (lBase is KArrayConst<*, *> && rBase is KArrayConst<*, *>) {
                // (= (const a) (const b)) ==> (= a b)
                checks += mkEq(lBase.value as KExpr<KSort>, rBase.value.uncheckedCast())
            }
            for (store in (lStores + rStores)) {
                val idx = store.index as KExpr<KSort>
                checks += lhs.select(idx) eq rhs.select(idx)
            }
            return rewrite(mkAnd(checks))
        }

        return mkEq(lhs, rhs)
    }

    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        simplifyApp(
            expr = expr as KApp<KArraySort<D, R>, KExpr<KSort>>,
            preprocess = { flatStores(expr) }
        ) {
            error("Always preprocessed")
        }

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
                    var allParentIndicesAreDistinct = true

                    /**
                     * Since all constants are trivially comparable we can guarantee, that
                     * all parents are distinct.
                     * Otherwise, we need to perform full check of parent indices.
                     * */
                    if (!index.definitelyIsConstant || storedIndexPosition < lastNonConstantIndex) {
                        /**
                         *  If non-constant index is among the indices we need to check,
                         *  we can check it first. Since non-constant indices are usually
                         *  not definitely distinct, we will not check all other indices.
                         * */
                        if (lastNonConstantIndex > storedIndexPosition
                            && !areDefinitelyDistinct(index, simplifiedIndices[lastNonConstantIndex])
                        ) {
                            allParentIndicesAreDistinct = false
                        }

                        var checkIdx = storedIndexPosition + 1
                        while (allParentIndicesAreDistinct && checkIdx < simplifiedIndices.size) {
                            if (!areDefinitelyDistinct(index, simplifiedIndices[checkIdx])) {
                                // possibly equal index, we can't squash stores
                                allParentIndicesAreDistinct = false
                            }
                            checkIdx++
                        }
                    }

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

    private val KExpr<*>.definitelyIsConstant: Boolean
        get() = this is KBitVecValue<*>
                || this is KFpValue<*>
                || this is KIntNumExpr
                || this is KRealNumExpr
                || this is KTrue
                || this is KFalse

    @Suppress("UNCHECKED_CAST")
    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        simplifyApp(expr as KApp<R, KExpr<KSort>>) { (array, index) ->
            var selectBaseArray = array as KExpr<KArraySort<*, *>>

            val (stores, _) = expandStores(selectBaseArray)

            for (store in stores) {
                // (select (store i v) i) ==> v
                if (store.index == index) {
                    return@simplifyApp store.value.uncheckedCast()
                }

                if (areDefinitelyDistinct(index, store.index.uncheckedCast())) {
                    // (select (store a i v) j), i != j ==> (select a j)
                    selectBaseArray = store.array as KExpr<KArraySort<*, *>>
                } else {
                    // possibly equal index, we can't expand stores
                    break
                }
            }

            // (select (const v) i) ==> v
            if (selectBaseArray is KArrayConst<*, *>) {
                return@simplifyApp selectBaseArray.value.uncheckedCast()
            }

            mkArraySelect(selectBaseArray.uncheckedCast(), index)
        }

    private fun <T : KArraySort<*, *>> expandStores(
        expr: KExpr<T>
    ): Pair<List<KArrayStore<*, *>>, KExpr<T>> {
        val stores = arrayListOf<KArrayStore<*, *>>()
        var base = expr
        while (base is KArrayStore<*, *>) {
            stores += base
            base = base.array as KExpr<T>
        }
        return stores to base
    }

    private fun <D : KSort, R : KSort> flatStores(expr: KExpr<KArraySort<D, R>>): SimplifierFlatArrayStoreExpr<D, R> {
        val stores = arrayListOf<KArrayStore<D, R>>()
        var base = expr
        while (base is KArrayStore<D, R>) {
            stores += base
            base = base.array
        }
        return SimplifierFlatArrayStoreExpr(
            ctx = ctx,
            base = base,
            indices = stores.map { it.index },
            values = stores.map { it.value }
        )
    }

    private class SimplifierFlatArrayStoreExpr<D : KSort, R : KSort>(
        ctx: KContext,
        val base: KExpr<KArraySort<D, R>>,
        val indices: List<KExpr<D>>,
        val values: List<KExpr<R>>
    ) : KApp<KArraySort<D, R>, KExpr<KSort>>(ctx) {

        override val args: List<KExpr<KSort>> = (listOf(base) + indices + values).uncheckedCast()

        override val decl: KDecl<KArraySort<D, R>>
            get() = ctx.mkArrayStoreDecl(base.sort)

        override val sort: KArraySort<D, R>
            get() = base.sort

        override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> {
            transformer as KArrayExprSimplifier
            return transformer.transform(this)
        }
    }

}
