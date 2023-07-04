package io.ksmt.solver.maxsat

import io.ksmt.solver.KSolverStatus

/**
 * @property satSoftConstraints
 * - MaxSAT has succeeded -> contains soft constraints from MaxSAT solution.
 * - MaxSAT has not succeeded -> contains soft constraints algorithm considered as satisfiable (incomplete solution).
 *
 * @property hardConstraintsSATStatus
 * Shows satisfiability status of hardly asserted constraints' conjunction.
 *
 * @property maxSATSucceeded
 * Shows whether MaxSAT calculation has succeeded or not.
 *
 * It may end without success in case of exceeding the timeout or in case solver started returning UNKNOWN during
 * MaxSAT calculation.
 *
 * @property timeoutExceeded
 * Shows whether timeout has been exceeded or not.
 */
class KMaxSATResult(
    val satSoftConstraints: List<SoftConstraint>,
    val hardConstraintsSATStatus: KSolverStatus,
    val maxSATSucceeded: Boolean,
    val timeoutExceeded: Boolean,
)
