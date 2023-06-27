package io.ksmt.solver.maxsat

import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort

class SoftConstraint(override val constraint: KExpr<KBoolSort>, val weight: Int) : Constraint
