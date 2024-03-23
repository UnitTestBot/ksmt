package io.ksmt.solver.maxsmt.test.statistics

import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.solver.maxsmt.test.utils.Solver

internal data class MaxSMTTestStatistics(val name: String, var smtSolver: Solver) {
    var maxSMTCallStatistics: KMaxSMTStatistics? = null
    var passed = false
    var ignoredTest = false
    var failedOnParsingOrConvertingExpressions = false
    var exceptionMessage: String? = null
    var checkedSoftConstraintsSumIsWrong = false
}
