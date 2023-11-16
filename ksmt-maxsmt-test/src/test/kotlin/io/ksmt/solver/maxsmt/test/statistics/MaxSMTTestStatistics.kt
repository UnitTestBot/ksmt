package io.ksmt.solver.maxsmt.test.statistics

import io.ksmt.solver.maxsmt.statistics.KMaxSMTStatistics

internal data class MaxSMTTestStatistics(val maxSMTCallStatistics: KMaxSMTStatistics) {
    var name = ""
    var maxSMTCallElapsedTimeMs: Long = 0
    var passed = false
    var correctnessError = true
}
