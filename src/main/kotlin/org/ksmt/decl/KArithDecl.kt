package org.ksmt.decl

import org.ksmt.expr.KArithExpr
import org.ksmt.sort.KArithSort

abstract class KArithDecl : KDecl<KArithExpr>() {
    override val sort = KArithSort
}
