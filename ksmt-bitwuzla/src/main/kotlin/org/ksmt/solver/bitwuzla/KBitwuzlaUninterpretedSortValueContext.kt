package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
import org.ksmt.sort.KUninterpretedSort

class KBitwuzlaUninterpretedSortValueContext(private val ctx: KContext) {
    private val sortsUniverses = hashMapOf<KUninterpretedSort, MutableMap<KExpr<*>, KExpr<KUninterpretedSort>>>()

    fun mkValue(sort: KUninterpretedSort, value: KBitVecValue<*>): KExpr<KUninterpretedSort> {
        val sortUniverse = sortsUniverses.getOrPut(sort) { hashMapOf() }
        return sortUniverse.getOrPut(value) {
            ctx.mkFreshConst("value!${sortUniverse.size}", sort)
        }
    }

    fun currentSortUniverse(sort: KUninterpretedSort): Set<KExpr<KUninterpretedSort>> =
        sortsUniverses[sort]?.map { it.value }?.toSet() ?: emptySet()
}
