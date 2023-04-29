package io.ksmt.expr.printer

import io.ksmt.expr.KExpr

interface ExpressionPrinter {
    /**
     * Append string as in StringBuilder.
     * */
    fun append(str: String)

    /**
     * Append an expression.
     * */
    fun append(expr: KExpr<*>)
}
