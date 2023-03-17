package org.ksmt.solver.fixtures.yices

import org.ksmt.expr.KExpr
import org.ksmt.solver.yices.KYicesContext
import org.ksmt.solver.yices.YicesTerm

class KTestYicesContext: KYicesContext() {
    override fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> YicesTerm): YicesTerm =
        internalize(expressions, expr, internalizer)

    private inline fun <K, V> internalize(
        cache: MutableMap<K, V>,
        key: K,
        internalizer: (K) -> V
    ): V = cache.getOrPut(key) {
        internalizer(key)
    }
}
