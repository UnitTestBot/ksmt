package io.ksmt.runner.core

import com.jetbrains.rd.framework.util.synchronizeWith
import com.jetbrains.rd.util.lifetime.Lifetime
import com.jetbrains.rd.util.lifetime.LifetimeDefinition
import com.jetbrains.rd.util.reactive.IScheduler
import com.jetbrains.rd.util.reactive.ISource
import kotlinx.coroutines.CompletableDeferred
import kotlinx.coroutines.Deferred

const val MAIN_PROCESS_NAME = "main"
const val CHILD_PROCESS_NAME = "child"

fun <T> IScheduler.pumpAsync(lifetime: Lifetime, block: () -> T): Deferred<T> {
    val ldef = lifetime.createNested()
    val deferred = CompletableDeferred<T>()

    ldef.synchronizeWith(deferred)

    invokeOrQueue {
        deferred.complete(block())
    }

    return deferred
}

fun <T> ISource<T>.adviseForConditionAsync(lifetime: Lifetime, condition: (T) -> Boolean): Deferred<Unit> {
    val ldef = lifetime.createNested()
    val deferred = CompletableDeferred<Unit>()

    ldef.synchronizeWith(deferred)

    advise(ldef) {
        if (condition(it)) {
            deferred.complete(Unit)
        }
    }

    return deferred
}

fun ISource<Boolean>.adviseForConditionAsync(lifetime: Lifetime): Deferred<Unit> =
    adviseForConditionAsync(lifetime) { it }

@Suppress("TooGenericExceptionCaught")
inline fun <T> LifetimeDefinition.terminateOnException(block: (Lifetime) -> T): T {
    try {
        return block(this)
    } catch (e: Throwable) {
        terminate()
        throw e
    }
}

suspend fun Lifetime.awaitTermination() {
    val deferred = CompletableDeferred<Unit>()
    onTermination { deferred.complete(Unit) }
    deferred.await()
}
