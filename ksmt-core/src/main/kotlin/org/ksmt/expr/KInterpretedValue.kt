package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KSort

/**
 * Specify that the expression is an interpreted value in some theory.
 * */
abstract class KInterpretedValue<T : KSort>(ctx: KContext) : KApp<T, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>>
        get() = emptyList()
}
