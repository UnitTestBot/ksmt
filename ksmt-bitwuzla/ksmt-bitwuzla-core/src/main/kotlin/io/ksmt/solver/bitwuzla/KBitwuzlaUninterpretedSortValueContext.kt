package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.sort.KUninterpretedSort

class KBitwuzlaUninterpretedSortValueContext(private val ctx: KContext) {
    private val sortsUniverses = hashMapOf<KUninterpretedSort, MutableMap<KExpr<*>, KUninterpretedSortValue>>()

    fun mkValue(sort: KUninterpretedSort, value: KBitVec32Value): KUninterpretedSortValue {
        val sortUniverse = sortsUniverses.getOrPut(sort) { hashMapOf() }
        return sortUniverse.getOrPut(value) {
            ctx.mkUninterpretedSortValue(sort, value.intValue)
        }
    }

    fun currentSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue> =
        sortsUniverses[sort]?.map { it.value }?.toSet() ?: emptySet()
}
