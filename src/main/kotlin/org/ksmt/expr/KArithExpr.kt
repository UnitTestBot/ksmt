package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArithSort

abstract class KArithExpr<A : KExpr<*>>(decl: KDecl<KArithSort>, args: List<A>) : KApp<KArithSort, A>(decl, args) {
    override val sort = KArithSort
}
