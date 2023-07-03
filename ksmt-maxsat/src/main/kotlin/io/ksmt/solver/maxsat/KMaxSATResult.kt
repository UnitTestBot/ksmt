package io.ksmt.solver.maxsat

import io.ksmt.solver.KSolverStatus

class KMaxSATResult(val satSoftConstraints: List<SoftConstraint>, val hardConstraintsSATStatus: KSolverStatus,
                    val maxSATSucceeded: Boolean, val timeoutExceeded: Boolean)
