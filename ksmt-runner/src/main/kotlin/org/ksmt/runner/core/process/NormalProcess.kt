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
        fun start(entrypoint: KClass<*>, args: List<String>, jvmArgs: List<String> = emptyList()): ProcessWrapper {
            val classPath = System.getProperty("java.class.path") ?: error("No class path")
            val entrypointClassName = entrypoint.qualifiedName ?: error("Entrypoint class name is not available")
            val javaHome = System.getProperty("java.home")
            val javaExecutable = Path(javaHome).resolve("bin").resolve("java")
            val workerCommand = listOf(
                javaExecutable.toAbsolutePath().toString(),
            ) + jvmArgs + listOf(
                "-classpath", classPath,
            ) + listOf(
                entrypointClassName
            ) + args
            val pb = ProcessBuilder(workerCommand).inheritIO()
            val process = pb.start()
            return NormalProcess(process)
        }

        fun startWithDebugger(entrypoint: KClass<*>, args: List<String>): ProcessWrapper =
            start(
                entrypoint,
                args,
                jvmArgs = listOf(
                    "-agentlib:jdwp=transport=dt_socket,server=y,suspend=y,quiet=n,address=*:5008"
                )
            )
    }
}
