package org.ksmt.expr

import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KSort

abstract class KExpr<T : KSort> {
    abstract val sort: T

    private val hash by lazy(LazyThreadSafetyMode.NONE) { hash() }
    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = hash
    abstract fun hash(): Int
    abstract fun equalTo(other: KExpr<*>): Boolean
    abstract fun accept(transformer: KTransformer): KExpr<T>
}
