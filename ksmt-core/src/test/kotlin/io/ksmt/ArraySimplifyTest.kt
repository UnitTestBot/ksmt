package io.ksmt

import io.ksmt.expr.KExpr
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KIntSort
import kotlin.test.Test


class ArraySimplifyTest: ExpressionSimplifyTest() {

    @Test
    fun testArraySelect() = runTest({ mkArraySort(intSort, intSort) }) { sort, runner ->
        generateArray(
            sort,
            GENERATED_ARRAY_SIZE,
            { a, (i), v -> mkArrayStoreNoSimplify(a, i, v) }
        ) { array, indices ->
            for ((index) in indices) {
                runner.check(
                    unsimplifiedExpr = mkArraySelectNoSimplify(array, index),
                    simplifiedExpr = mkArraySelect(array, index),
                    printArgs = { "$index | $array" }
                )
            }
        }
    }

    @Test
    fun testArray2Select() = runTest({ mkArraySort(intSort, intSort, intSort) }) { sort, runner ->
        generateArray(
            sort,
            GENERATED_ARRAY_SIZE,
            { a, (i0, i1), v -> mkArrayStoreNoSimplify(a, i0, i1, v) }
        ) { array, indices ->
            for ((i0, i1) in indices) {
                runner.check(
                    unsimplifiedExpr = mkArraySelectNoSimplify(array, i0, i1),
                    simplifiedExpr = mkArraySelect(array, i0, i1),
                    printArgs = { "$i0, $i1 | $array" }
                )
            }
        }
    }

    @Test
    fun testArray3Select() = runTest({ mkArraySort(intSort, intSort, intSort, intSort) }) { sort, runner ->
        generateArray(
            sort,
            GENERATED_ARRAY_SIZE,
            { a, (i0, i1, i2), v -> mkArrayStoreNoSimplify(a, i0, i1, i2, v) }
        ) { array, indices ->
            for ((i0, i1, i2) in indices) {
                runner.check(
                    unsimplifiedExpr = mkArraySelectNoSimplify(array, i0, i1, i2),
                    simplifiedExpr = mkArraySelect(array, i0, i1, i2),
                    printArgs = { "$i0, $i1, $i2 | $array" }
                )
            }
        }
    }

    @Test
    fun testArrayNSelect() = runTest({ mkArrayNSort(List(5) { intSort }, intSort) }) { sort, runner ->
        generateArray(
            sort,
            GENERATED_ARRAY_SIZE,
            { a, idx, v -> mkArrayNStoreNoSimplify(a, idx, v) }
        ) { array, indices ->
            for (idx in indices) {
                runner.check(
                    unsimplifiedExpr = mkArrayNSelectNoSimplify(array, idx),
                    simplifiedExpr = mkArrayNSelect(array, idx),
                    printArgs = { "$idx | $array" }
                )
            }
        }
    }

    private inline fun <A : KArraySortBase<KIntSort>> KContext.generateArray(
        sort: A,
        size: Int,
        mkStore: KContext.(KExpr<A>, List<KExpr<KIntSort>>, KExpr<KIntSort>) -> KExpr<A>,
        analyze: (KExpr<A>, List<List<KExpr<KIntSort>>>) -> Unit
    ) {
        val indices = arrayListOf<List<KExpr<KIntSort>>>()
        var array: KExpr<A> = mkConst("array", sort)

        for (i in 0 until size) {
            val storeIndices = List(sort.domainSorts.size) { idx ->
                when {
                    random.nextDouble() < UNINTERPRETED_INDEX_PROBABILITY ->
                        mkFreshConst("idx", intSort)

                    random.nextDouble() < DUPLICATE_INDEX_PROBABILITY ->
                        mkIntNum(random.nextInt(0, i / 2 + 1))

                    else -> mkIntNum(i + idx)
                }
            }

            array = mkStore(array, storeIndices, mkIntNum(i))
            indices.add(storeIndices)

            analyze(array, indices)
        }
    }

    companion object {
        private const val GENERATED_ARRAY_SIZE = 100
        private const val UNINTERPRETED_INDEX_PROBABILITY = 0.1
        private const val DUPLICATE_INDEX_PROBABILITY = 0.2
    }
}
