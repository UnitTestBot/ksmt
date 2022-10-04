package org.ksmt.runner.core

import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.reactive.IScheduler
import com.jetbrains.rd.util.reactive.ISource
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred

const val MAIN_PROCESS_NAME = "main"
const val CHILD_PROCESS_NAME = "child"

fun <T> IScheduler.pumpAsync(lifetime: Lifetime, block: () -> T): Deferred<T> {
    val ldef = lifetime.createNested()
    val deferred = CompletableDeferred<T>()

    ldef.onTermination { deferred.cancel() }
    deferred.invokeOnCompletion { ldef.terminate() }

    invokeOrQueue {
        deferred.complete(block())
    }

    return deferred
}

fun <T> ISource<T>.adviseForConditionAsync(lifetime: Lifetime, condition: (T) -> Boolean): Deferred<Unit> {
    val ldef = lifetime.createNested()
    val deferred = CompletableDeferred<Unit>()

    ldef.onTermination { deferred.cancel() }
    deferred.invokeOnCompletion { ldef.terminate() }

    advise(ldef) {
        if (condition(it)) {
            deferred.complete(Unit)
        }
    }

    return deferred
}

fun ISource<Boolean>.adviseForConditionAsync(lifetime: Lifetime): Deferred<Unit> =
    adviseForConditionAsync(lifetime) { it }
