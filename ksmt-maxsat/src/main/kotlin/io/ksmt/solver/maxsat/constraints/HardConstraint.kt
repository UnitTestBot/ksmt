package io.ksmt.solver.maxsat.constraints

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class HardConstraint(override val expression: KExpr<KBoolSort>) : Constraint
