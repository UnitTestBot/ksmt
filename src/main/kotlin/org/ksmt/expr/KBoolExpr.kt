package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

abstract class KBoolExpr<A: KExpr<*>>(decl: KDecl<KBoolSort>, args: List<A>) : KApp<KBoolSort, A>(decl, args) {
    override val sort = KBoolSort
}
