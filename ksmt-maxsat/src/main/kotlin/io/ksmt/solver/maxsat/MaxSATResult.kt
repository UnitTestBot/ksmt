package io.ksmt.solver.maxsat

import io.ksmt.solver.KSolverStatus

class MaxSATResult(val satSoftConstraints: List<SoftConstraint>,
                   val hardConstraintsSATStatus: KSolverStatus, val maxSATSucceeded: Boolean)
