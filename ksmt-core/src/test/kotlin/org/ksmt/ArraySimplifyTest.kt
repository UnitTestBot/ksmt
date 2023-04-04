package org.ksmt

import org.ksmt.expr.KExpr
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KIntSort
import kotlin.test.Test


class ArraySimplifyTest: ExpressionSimplifyTest() {

    @Test
    fun testArraySelect() = runTest({ mkArraySort(intSort, intSort) }) { sort, runner ->
        generateArray(sort, GENERATED_ARRAY_SIZE) { array, indices ->
            for (index in indices) {
                runner.check(
                    unsimplifiedExpr = mkArraySelectNoSimplify(array, index),
                    simplifiedExpr = mkArraySelect(array, index),
                    printArgs = { "$index | $array" }
                )
            }
        }
    }

    private inline fun KContext.generateArray(
        sort: KArraySort<KIntSort, KIntSort>,
        size: Int,
        analyze: (KExpr<KArraySort<KIntSort, KIntSort>>, List<KExpr<KIntSort>>) -> Unit
    ) {
        val indices = arrayListOf<KExpr<KIntSort>>()
        var array = mkConst("array", sort)

        for (i in 0 until size) {
            val index = when {
                random.nextDouble() < UNINTERPRETED_INDEX_PROBABILITY -> mkFreshConst("idx", intSort)
                random.nextDouble() < DUPLICATE_INDEX_PROBABILITY -> mkIntNum(i / 2)
                else -> mkIntNum(i)
            }

            array = mkArrayStoreNoSimplify(array, index, mkIntNum(i))
            indices.add(index)

            analyze(array, indices)
        }
    }

    companion object {
        private const val GENERATED_ARRAY_SIZE = 100
        private const val UNINTERPRETED_INDEX_PROBABILITY = 0.1
        private const val DUPLICATE_INDEX_PROBABILITY = 0.1
    }
}
