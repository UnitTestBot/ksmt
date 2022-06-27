package org.ksmt.solver

import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort

interface KSolver : AutoCloseable {
    fun assert(expr: KExpr<KBoolSort>)
    fun check(): KSolverStatus
    fun model(): KModel
}
