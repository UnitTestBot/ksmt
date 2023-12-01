package io.ksmt.solver.maxsmt.test.statistics

import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics
import io.ksmt.solver.maxsmt.test.utils.Solver
import io.ksmt.solver.maxsmt.test.utils.Solver.Z3

internal data class MaxSMTTestStatistics(val maxSMTCallStatistics: KMaxSMTStatistics) {
    var smtSolver: Solver = Z3
    var name = ""
    var maxSMTCallElapsedTimeMs: Long = 0
    var passed = false
    var checkedSoftConstraintsSumIsWrong = false
}
