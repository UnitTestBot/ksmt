package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KSort

abstract class KExpr<T : KSort> {
    abstract fun KContext.sort(): T
    abstract fun accept(transformer: KTransformer): KExpr<T>

    //  Contexts guarantee that any two equivalent expressions will be the same kotlin object
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = System.identityHashCode(this)
}
