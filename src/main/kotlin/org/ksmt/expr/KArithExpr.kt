package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArithSort

abstract class KArithExpr : KExpr<KArithSort>() {
    override val sort = KArithSort
    abstract override val decl: KDecl<KArithSort>
}
