package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KSort

abstract class KExpr<T : KSort> {
    abstract fun KContext.sort(): T
    private val hash by lazy { hash() }
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = hash
    abstract fun hash(): Int
    abstract fun equalTo(other: KExpr<*>): Boolean
    abstract fun accept(transformer: KTransformer): KExpr<T>
}
