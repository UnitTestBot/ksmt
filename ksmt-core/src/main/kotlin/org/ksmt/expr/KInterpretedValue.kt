package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KSort

/**
 * Specify that the expression is an interpreted value in some theory.
 * */
abstract class KInterpretedValue<T : KSort>(ctx: KContext) : KApp<T, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = emptyList()
}
