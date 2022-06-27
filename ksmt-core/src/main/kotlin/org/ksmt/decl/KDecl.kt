package org.ksmt.decl

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KDecl<T : KSort>(
    ctx: KContext,
    val name: String,
    val sort: T
) : KAst(ctx) {
    abstract fun apply(args: List<KExpr<*>>): KApp<T, *>
    abstract fun <R> accept(visitor: KDeclVisitor<R>): R

    //  Contexts guarantee that any two equivalent declarations will be the same kotlin object
    override fun hashCode(): Int = System.identityHashCode(this)
    override fun equals(other: Any?): Boolean = this === other
}
