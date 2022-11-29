package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

abstract class KQuantifier(
    ctx: KContext,
    val body: KExpr<KBoolSort>,
    val bounds: List<KDecl<*>>,
) : KExpr<KBoolSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    abstract fun printQuantifierName(): String

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append('(')
        append(printQuantifierName())
        append('(')

        bounds.forEach { bound ->
            append('(')
            append(bound.name)
            append(' ')
            bound.sort.print(this)
            append(')')
        }

        append(')')
        body.print(this)
        append(')')
    }
}
