package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.sort.KSort

/**
 * Specify that the expression is an interpreted value in some theory.
 * */
abstract class KInterpretedValue<T : KSort>(ctx: KContext) : KApp<T, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = emptyList()
}
