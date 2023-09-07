package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.expr.KBitVec32Value
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.sort.KUninterpretedSort

class KBitwuzlaUninterpretedSortValueContext(private val ctx: KContext) {
    private val sortsUniverses = hashMapOf<KUninterpretedSort, MutableMap<KExpr<*>, KUninterpretedSortValue>>()

    fun mkValue(sort: KUninterpretedSort, value: KBitVec32Value): KUninterpretedSortValue {
        return registerValue(ctx.mkUninterpretedSortValue(sort, value.intValue))
    }

    fun registerValue(value: KUninterpretedSortValue): KUninterpretedSortValue {
        val sortsUniverse = sortsUniverses.getOrPut(value.sort) { hashMapOf() }
        return sortsUniverse.getOrPut(value) { value }
    }

    fun currentSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue> =
        sortsUniverses[sort]?.map { it.value }?.toSet() ?: emptySet()
}
