package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

abstract class KQuantifier(val body: KExpr<KBoolSort>, val bounds: List<KDecl<*>>) : KExpr<KBoolSort>() {
    override fun KContext.sort(): KBoolSort = mkBoolSort()
}
