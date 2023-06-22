package io.ksmt.solver.maxsmt

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

interface Constraint {
    val constraint: KExpr<KBoolSort>
}
