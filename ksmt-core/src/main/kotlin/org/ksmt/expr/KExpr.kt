package org.ksmt.expr

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort

abstract class KExpr<T : KSort>(ctx: KContext) : KAst(ctx) {
    abstract val sort: T

    @Deprecated("Use property access syntax", ReplaceWith("sort"))
    fun sort(): T = sort

    abstract fun accept(transformer: KTransformerBase): KExpr<T>

    //  Contexts guarantee that any two equivalent expressions will be the same kotlin object
    override fun equals(other: Any?): Boolean = this === other

    override fun hashCode(): Int = System.identityHashCode(this)

    open fun computeExprSort(): T = sort
    open fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {}
}
