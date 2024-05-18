package io.ksmt.solver.maxsmt

import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.maxsmt.constraints.SoftConstraint

/**
 * @property satSoftConstraints
 * - MaxSMT has succeeded -> contains soft constraints from MaxSMT solution.
 * - MaxSMT has not succeeded -> contains soft constraints algorithm considered as satisfiable (suboptimal solution).
 *
 * @property hardConstraintsSatStatus
 * Shows satisfiability status of hardly asserted constraints' conjunction.
 *
 * @property timeoutExceededOrUnknown
 * Shows whether timeout has exceeded, solver was interrupted or returned UNKNOWN (can happen when timeout has exceeded
 * or by some other reason).
 *
 * @property maxSMTSucceeded
 * Shows whether MaxSMT calculation has succeeded or not.
 *
 * It may end without success in case of exceeding the timeout or in case solver started returning UNKNOWN during
 * MaxSAT calculation.
 */
class KMaxSMTResult(
    val satSoftConstraints: List<SoftConstraint>,
    val hardConstraintsSatStatus: KSolverStatus,
    val timeoutExceededOrUnknown: Boolean,
    val maxSMTSucceeded: Boolean
)
