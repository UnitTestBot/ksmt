package org.ksmt

abstract class KAst(val ctx: KContext) {
    abstract fun print(): String
    override fun toString(): String = with(ctx) { stringRepr }
}
