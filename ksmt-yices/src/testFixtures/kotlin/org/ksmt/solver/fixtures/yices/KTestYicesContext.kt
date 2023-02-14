package org.ksmt.solver.fixtures.yices

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.yices.KYicesContext
import org.ksmt.solver.yices.YicesTerm
import org.ksmt.solver.yices.YicesSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KTestYicesContext: KYicesContext() {
    override fun internalizeExpr(expr: KExpr<*>, internalizer: (KExpr<*>) -> YicesTerm): YicesTerm =
        internalize(expressions, expr, internalizer)

    override fun internalizeSort(sort: KSort, internalizer: (KSort) -> YicesSort): YicesSort =
        internalize(sorts, sort, internalizer)

    override fun internalizeDecl(decl: KDecl<*>, internalizer: (KDecl<*>) -> YicesTerm): YicesTerm {
        return if (decl.sort is KArraySort<*, *>)
            super.internalizeDecl(decl, internalizer)
        else
            internalize(decls, decl, internalizer)
    }

    private inline fun <K, V> internalize(
        cache: MutableMap<K, V>,
        key: K,
        internalizer: (K) -> V
    ): V = cache.getOrPut(key) {
        internalizer(key)
    }
}