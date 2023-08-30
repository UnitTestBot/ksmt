package io.ksmt.solver.maxsat.constraints

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

interface Constraint {
    val expression: KExpr<KBoolSort>
}
