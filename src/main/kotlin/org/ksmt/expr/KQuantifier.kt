package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort
import java.util.Objects

abstract class KQuantifier(val body: KExpr<KBoolSort>, val bounds: List<KDecl<*>>) : KExpr<KBoolSort>() {
    override fun KContext.sort(): KBoolSort = mkBoolSort()
    override fun hash(): Int = Objects.hash(javaClass, body, bounds)
}
