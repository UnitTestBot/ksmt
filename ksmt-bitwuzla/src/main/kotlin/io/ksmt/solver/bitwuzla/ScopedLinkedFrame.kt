package io.ksmt.solver.bitwuzla

internal class ScopedLinkedFrame<T> private constructor(
    private var current: LinkedFrame<T>,
    private inline val createNewFrame: () -> T,
    private inline val copyFrame: (T) -> T
) {
    constructor(
        currentFrame: T,
        createNewFrame: () -> T,
        copyFrame: (T) -> T
    ) : this(LinkedFrame(currentFrame), createNewFrame, copyFrame)

    constructor(
        createNewFrame: () -> T,
        copyFrame: (T) -> T
    ) : this(createNewFrame(), createNewFrame, copyFrame)

    val currentFrame: T
        get() = current.value

    val currentScope: UInt
        get() = current.scope

    fun stacked(): ArrayDeque<T> = ArrayDeque<T>().also { stack ->
        forEachReversed { frame ->
            stack.addLast(frame)
        }
    }

    fun push() {
        current = LinkedFrame(createNewFrame(), current)
    }

    fun pop(n: UInt) {
        repeat(n.toInt()) {
            current = current.previous ?: throw IllegalStateException("Can't pop the bottom scope")
        }
        recreateTopFrame()
    }

    private fun recreateTopFrame() {
        val newTopFrame = copyFrame(currentFrame)
        current = LinkedFrame(newTopFrame, current.previous)
    }

    fun fork(parent: ScopedLinkedFrame<T>) {
        current = parent.current
        recreateTopFrame()
    }

    private inline fun forEachReversed(action: (T) -> Unit) {
        var cur: LinkedFrame<T>? = current
        while (cur != null) {
            action(cur.value)
            cur = cur.previous
        }
    }

    private class LinkedFrame<E>(
        val value: E,
        val previous: LinkedFrame<E>? = null
    ) {
        val scope: UInt = previous?.scope?.plus(1u) ?: 0u
    }

}
