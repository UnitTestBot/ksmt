package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.expr.KBitVec32Value
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.sort.KUninterpretedSort

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
