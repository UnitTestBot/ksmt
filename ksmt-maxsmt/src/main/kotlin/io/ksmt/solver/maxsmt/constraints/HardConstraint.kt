package io.ksmt.solver.maxsmt.constraints

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class HardConstraint(override val expression: KExpr<KBoolSort>) : Constraint
