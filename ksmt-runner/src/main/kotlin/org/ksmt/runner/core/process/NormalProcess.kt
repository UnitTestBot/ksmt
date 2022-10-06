package org.ksmt.runner.core.process

import kotlin.io.path.Path
import kotlin.reflect.KClass

class NormalProcess private constructor(private val process: Process) : ProcessWrapper {
    override val isAlive: Boolean
        get() = process.isAlive

    override fun destroyForcibly() {
        process.destroyForcibly()
    }

    companion object {
        fun start(entrypoint: KClass<*>, args: List<String>): ProcessWrapper {
            val classPath = System.getProperty("java.class.path") ?: error("No class path")
            val entrypointClassName = entrypoint.qualifiedName ?: error("Entrypoint class name is not available")
            val javaHome = System.getProperty("java.home")
            val javaExecutable = Path(javaHome).resolve("bin").resolve("java")
            val workerCommand = listOf(
                javaExecutable.toAbsolutePath().toString(),
                "-classpath", classPath,
            ) + listOf(
                entrypointClassName
            ) + args
            val pb = ProcessBuilder(workerCommand).inheritIO()
            val process = pb.start()
            return NormalProcess(process)
        }
    }
}
