package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KApp
import org.ksmt.expr.KArrayConst
import org.ksmt.expr.KArraySelect
import org.ksmt.expr.KArrayStore
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr

interface KArrayExprSimplifier : KExprSimplifierBase {

    @Suppress("UNCHECKED_CAST")
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
                checks += mkEq(lBase.value.asExpr(lhs.sort.range), rBase.value.asExpr(lhs.sort.range))
            }
            for (store in (lStores + rStores)) {
                val idx = store.index as KExpr<KSort>
                checks += lhs.select(idx) eq rhs.select(idx)
            }
            return mkAnd(checks)
        }

        // todo: array_rewriter.cpp:740
        return mkEq(lhs, rhs)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <D : KSort, R : KSort> transform(expr: KArrayStore<D, R>): KExpr<KArraySort<D, R>> =
        simplifyApp(expr as KApp<KArraySort<D, R>, KExpr<KSort>>) { (array, index, value) ->
            // (store (const v) i v) ==> (const v)
            if (array is KArrayConst<*, *> && array.value == value) {
                return@simplifyApp array.asExpr(expr.sort)
            }

            // (store a i (select a i)) ==> a
            if (value is KArraySelect<*, *> && array == value.array && index == value.index) {
                return@simplifyApp array.asExpr(expr.sort)
            }

            val (stores, _) = expandStores(array as KExpr<KArraySort<*, *>>)
            if (stores.isNotEmpty()) {
                val parentStores = arrayListOf<KArrayStore<*, *>>()
                for (store in stores) {
                    // (store (store a i x) i y) ==> (store a i y)
                    if (store.index == index) {
                        var base = store.array as KExpr<KArraySort<*, *>>
                        parentStores.asReversed().forEach {
                            base = mkArrayStore(base, it.index as KExpr<KSort>, it.value as KExpr<KSort>)
                        }
                        return@simplifyApp mkArrayStore(base, index, value).asExpr(expr.sort)
                    }

                    if (areDefinitelyDistinct(index, store.index.asExpr(index.sort))) {
                        // (store (store a i x) j y), i != j ==> (store (store a j y) i x)
                        parentStores += store
                    } else {
                        // possibly equal index, we can't squash stores
                        break
                    }
                }
            }

            mkArrayStore(array.asExpr(expr.sort), index, value).asExpr(expr.sort)
        }

    @Suppress("UNCHECKED_CAST")
    override fun <D : KSort, R : KSort> transform(expr: KArraySelect<D, R>): KExpr<R> =
        simplifyApp(expr as KApp<R, KExpr<KSort>>) { (array, index) ->
            var selectBaseArray = array as KExpr<KArraySort<*, *>>

            val (stores, _) = expandStores(selectBaseArray)

            for (store in stores) {
                // (select (store i v) i) ==> v
                if (store.index == index) {
                    return@simplifyApp store.value.asExpr(expr.sort)
                }

                if (areDefinitelyDistinct(index, store.index.asExpr(index.sort))) {
                    // (select (store a i v) j), i != j ==> (select a j)
                    selectBaseArray = store.array as KExpr<KArraySort<*, *>>
                } else {
                    // possibly equal index, we can't expand stores
                    break
                }
            }

            // (select (const v) i) ==> v
            if (selectBaseArray is KArrayConst<*, *>) {
                return@simplifyApp selectBaseArray.value.asExpr(expr.sort)
            }

            // todo: array_rewriter.cpp:199

            mkArraySelect(selectBaseArray.asExpr(expr.array.sort), index)
        }

    @Suppress("UNCHECKED_CAST")
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

}
