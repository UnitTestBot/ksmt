package io.ksmt.solver.maxsmt.utils

import kotlin.time.Duration
import kotlin.time.TimeSource.Monotonic.ValueTimeMark

internal object TimerUtils {
    fun computeRemainingTime(timeout: Duration, markStart: ValueTimeMark): Duration {
        return timeout - markStart.elapsedNow()
    }

    fun timeoutExceeded(timeout: Duration): Boolean =
        timeout.isNegative() || timeout == Duration.ZERO

    fun timeoutExceeded(timeout: Duration, markStart: ValueTimeMark): Boolean {
        val remainingTime = computeRemainingTime(timeout, markStart)
        return timeoutExceeded(remainingTime)
    }
}
