package org.ksmt.expr

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KSort

abstract class KExpr<T : KSort>(ctx: KContext): KAst(ctx) {
    abstract fun sort(): T
    abstract fun accept(transformer: KTransformer): KExpr<T>

    //  Contexts guarantee that any two equivalent expressions will be the same kotlin object
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}
