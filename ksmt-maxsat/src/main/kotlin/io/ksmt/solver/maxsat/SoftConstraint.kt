package io.ksmt.solver.maxsat

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class SoftConstraint(val expression: KExpr<KBoolSort>, val weight: Int)
