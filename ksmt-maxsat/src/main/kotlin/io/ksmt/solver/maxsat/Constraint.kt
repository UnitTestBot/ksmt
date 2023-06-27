package io.ksmt.solver.maxsat

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

interface Constraint {
    val constraint: KExpr<KBoolSort>
}
