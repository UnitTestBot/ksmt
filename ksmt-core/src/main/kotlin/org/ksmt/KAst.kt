package org.ksmt

import org.ksmt.cache.CustomObjectEquality

abstract class KAst(val ctx: KContext) : CustomObjectEquality {
    abstract fun print(builder: StringBuilder)

    override fun toString(): String = with(ctx) { stringRepr }

    abstract override fun customEquals(other: Any): Boolean

    abstract override fun customHashCode(): Int
}
