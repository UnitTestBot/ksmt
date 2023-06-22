package io.ksmt.solver.maxsmt

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class HardConstraint(override val constraint: KExpr<KBoolSort>) : Constraint
