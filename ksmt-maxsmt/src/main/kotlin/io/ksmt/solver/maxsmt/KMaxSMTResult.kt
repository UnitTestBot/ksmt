package io.ksmt.solver.maxsmt

import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.maxsmt.constraints.SoftConstraint

/**
 * @property satSoftConstraints
 * Contains soft constraints algorithm considered as the best solution found by the moment.
 *
 * @property hardConstraintsSatStatus
 * Shows satisfiability status of hardly asserted constraints' conjunction.
 *
 * @property timeoutExceededOrUnknown
 * Shows whether timeout has exceeded, solver was interrupted or returned UNKNOWN (can happen when timeout has exceeded
 * or by some other reason).
 *
 * It may end without success in case of exceeding the timeout or in case solver started returning UNKNOWN during
 * MaxSAT calculation.
 */
class KMaxSMTResult(
    val satSoftConstraints: List<SoftConstraint>,
    val hardConstraintsSatStatus: KSolverStatus,
    val timeoutExceededOrUnknown: Boolean
)
