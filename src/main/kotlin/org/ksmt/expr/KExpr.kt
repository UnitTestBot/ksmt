package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KSort

interface KExpr<T : KSort> {
    fun KContext.sort(): T
    fun accept(transformer: KTransformer): KExpr<T>

    /* Contexts guarantee that any two equivalent expressions will be the same kotlin object.
    *  Therefore, we don't need to override equals and hashCode
    *  */
}
