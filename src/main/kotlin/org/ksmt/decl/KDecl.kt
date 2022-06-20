package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import java.util.*

abstract class KDecl<T : KSort>(val name: String, val sort: T) {
    abstract fun apply(args: List<KExpr<*>>): KExpr<T>
    abstract fun <R> accept(visitor: KDeclVisitor<R>): R
    override fun hashCode(): Int = Objects.hash(name, sort)
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false
        other as KDecl<*>
        if (name != other.name) return false
        if (sort != other.sort) return false
        return true
    }
}
