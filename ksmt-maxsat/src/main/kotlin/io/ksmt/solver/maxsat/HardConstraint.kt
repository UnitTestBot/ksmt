package io.ksmt.solver.maxsat

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class HardConstraint(override val expression: KExpr<KBoolSort>) : Constraint
