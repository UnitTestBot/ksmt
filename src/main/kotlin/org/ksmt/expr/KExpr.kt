package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort
import java.util.*

abstract class KExpr<T : KSort<T>> {
    abstract val sort: T
    abstract val decl: KDecl<T>
    abstract val args: List<KExpr<*>>

    private val hash by lazy(LazyThreadSafetyMode.NONE) { Objects.hash(javaClass, sort, decl, args) }

    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = hash

    internal fun equalTo(other: KExpr<*>): Boolean {
        if (javaClass != other.javaClass) return false
        if (decl != other.decl) return false
        if (sort != other.sort) return false
        if (args != other.args) return false
        return true
    }

}
