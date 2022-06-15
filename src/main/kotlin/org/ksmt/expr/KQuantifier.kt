package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort
import java.util.Objects

abstract class KQuantifier(val body: KExpr<KBoolSort>, val bounds: List<KDecl<*>>) : KExpr<KBoolSort>() {
    override val sort: KBoolSort = KBoolSort

    override fun hash(): Int = Objects.hash(javaClass, sort, body, bounds)

    override fun equalTo(other: KExpr<*>): Boolean {
        if (this === other) return true
        if (javaClass != other.javaClass) return false
        other as KQuantifier
        if (body != other.body) return false
        if (bounds != other.bounds) return false
        return true
    }
}
