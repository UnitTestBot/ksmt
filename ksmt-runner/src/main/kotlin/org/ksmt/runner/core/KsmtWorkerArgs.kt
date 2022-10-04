package org.ksmt.runner.core

import kotlin.time.Duration
import kotlin.time.Duration.Companion.milliseconds

open class KsmtWorkerArgs(
    val id: Int,
    val port: Int,
    val workerTimeout: Duration
) {
    constructor(other: KsmtWorkerArgs) : this(other.id, other.port, other.workerTimeout)
    constructor(data: Map<String, String>) : this(
        id = data.getValue("id").toInt(),
        port = data.getValue("port").toInt(),
        workerTimeout = data.getValue("workerTimeout").toLong().milliseconds,
    )

    open fun MutableMap<String, String>.addArgs() {
        put("id", "$id")
        put("port", "$port")
        put("workerTimeout", "${workerTimeout.inWholeMilliseconds}")
    }

    fun toMap(): Map<String, String> = buildMap { addArgs() }
    fun toList(): List<String> = toMap().flatMap { it.toPair().toList() }

    companion object {
        fun argsMap(data: List<String>): Map<String, String> =
            data.windowed(size = 2, step = 2) { Pair(it.first(), it.last()) }.toMap()
        fun fromList(data: List<String>) = KsmtWorkerArgs(argsMap(data))
    }
}
