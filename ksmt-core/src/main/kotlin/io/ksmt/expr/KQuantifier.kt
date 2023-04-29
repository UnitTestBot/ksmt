package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.printer.ExpressionPrinter
import io.ksmt.sort.KBoolSort

abstract class KQuantifier(
    ctx: KContext,
    val body: KExpr<KBoolSort>,
    val bounds: List<KDecl<*>>,
) : KExpr<KBoolSort>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    abstract fun printQuantifierName(): String

    override fun print(printer: ExpressionPrinter) {
        val str = buildString {
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

            appendLine(')')
            body.print(this)
            appendLine()
            append(')')
        }
        printer.append(str)
    }
}
