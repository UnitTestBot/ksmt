package org.ksmt

abstract class KAst(val ctx: KContext) {
    abstract fun print(builder: StringBuilder)

    override fun toString(): String = with(ctx) { stringRepr }
}
