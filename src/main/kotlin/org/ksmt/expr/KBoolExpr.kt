package org.ksmt.expr

import org.ksmt.sort.KBoolSort

abstract class KBoolExpr<A: KExpr<*>>(args: List<A>) : KApp<KBoolSort, A>(args) {
    override val sort = KBoolSort
}
