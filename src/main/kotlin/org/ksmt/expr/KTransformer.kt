package org.ksmt.expr

import org.ksmt.sort.KBoolSort

class KTransformer{
    fun visit(expr: KExpr<*>): Nothing = TODO()
    fun visit(expr: KAndExpr): KExpr<KBoolSort>{
        TODO()
    }
}