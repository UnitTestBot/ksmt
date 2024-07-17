package io.ksmt.solver.maxsmt.statistics

import io.ksmt.solver.maxsmt.KMaxSMTContext

data class KMaxSMTStatistics(val maxSmtCtx: KMaxSMTContext) {
    var timeoutMs: Long = 0
    var elapsedTimeMs: Long = 0
    var timeInSolverQueriesMs: Long = 0
    var queriesToSolverNumber = 0
}
