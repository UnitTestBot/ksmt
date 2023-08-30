package io.ksmt.solver.maxsat.utils

import kotlin.time.Duration
import kotlin.time.DurationUnit
import kotlin.time.toDuration

internal object TimerUtils {
    fun computeRemainingTime(timeout: Duration, clockStart: Long): Duration {
        val msUnit = DurationUnit.MILLISECONDS
        return timeout - (System.currentTimeMillis().toDuration(msUnit) - clockStart.toDuration(msUnit))
    }
}
