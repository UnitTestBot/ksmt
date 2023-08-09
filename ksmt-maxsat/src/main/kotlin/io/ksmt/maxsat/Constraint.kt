package io.ksmt.maxsat

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

interface Constraint {
    val expression: KExpr<KBoolSort>
}
