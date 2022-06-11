package org.ksmt.decl

import org.ksmt.expr.KBoolExpr
import org.ksmt.sort.KBoolSort

abstract class KBoolDecl : KDecl<KBoolExpr>() {
    override val sort = KBoolSort
}
