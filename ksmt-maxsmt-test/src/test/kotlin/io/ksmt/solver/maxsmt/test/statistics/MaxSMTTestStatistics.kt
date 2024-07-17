package io.ksmt.solver.maxsmt.test.statistics

import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.solver.maxsmt.test.utils.Solver

internal data class MaxSMTTestStatistics(val name: String, var smtSolver: Solver) {
    var maxSMTCallStatistics: KMaxSMTStatistics? = null
    var passed = false
    var ignoredTest = false
    var failedOnParsingOrConvertingExpressions = false
    var exceptionMessage: String? = null
    var elapsedTimeMs: Long = 0
    /**
     * It's false when a sum is more than optimal in case of SubOpt
     * or is different from expected in case of Opt.
     */
    var checkedSoftConstraintsSumIsWrong = false
    var optimalWeight: ULong = 0U
    var foundSoFarWeight: ULong = 0U
    /**
     * Shows whether timeout has been exceeded, solver was terminated or returned UNKNOWN.
     */
    var timeoutExceededOrUnknown: Boolean = true
}
