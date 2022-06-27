package org.ksmt.decl

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import java.util.*

abstract class KDecl<T : KSort>(
    ctx: KContext,
    val name: String,
    val sort: T
): KAst(ctx) {
    abstract fun KContext.apply(args: List<KExpr<*>>): KApp<T, *>
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
