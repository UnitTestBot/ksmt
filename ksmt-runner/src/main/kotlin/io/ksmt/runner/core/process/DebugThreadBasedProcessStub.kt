package io.ksmt.runner.core.process

import kotlin.reflect.KClass

@Suppress("unused")
class DebugThreadBasedProcessStub private constructor(private val thread: Thread) : ProcessWrapper {
    override val isAlive: Boolean
        get() = thread.isAlive

    override fun destroyForcibly() {
        thread.interrupt()
    }

    companion object {
        fun start(entrypoint: KClass<*>, args: List<String>): ProcessWrapper {
            val mainMethod = entrypoint.java.declaredMethods.first { it.name == "main" }
            val thread = Thread {
                mainMethod.invoke(null, args.toTypedArray())
            }
            thread.start()
            return DebugThreadBasedProcessStub(thread)
        }
    }
}
