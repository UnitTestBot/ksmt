package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

abstract class KBoolExpr : KExpr<KBoolSort>() {
    override val sort = KBoolSort
    abstract override val decl: KDecl<KBoolSort>
}
