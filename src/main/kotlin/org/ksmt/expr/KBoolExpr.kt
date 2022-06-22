package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KBoolSort

abstract class KBoolExpr<A: KExpr<*>>(args: List<A>) : KApp<KBoolSort, A>(args) {
    override fun KContext.sort(): KBoolSort = mkBoolSort()
}
