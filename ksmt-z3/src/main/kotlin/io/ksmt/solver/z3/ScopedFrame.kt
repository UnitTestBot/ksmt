package io.ksmt.solver.z3

import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap

internal interface ScopedFrame<T> {
    val currentScope: UInt
    val currentFrame: T

    fun push()
    fun pop(n: UInt = 1u)
}

internal class ScopedArrayFrameOfLong2ObjectOpenHashMap<V>(
    currentFrame: Long2ObjectOpenHashMap<V>
) : ScopedFrame<Long2ObjectOpenHashMap<V>> {
    constructor() : this(Long2ObjectOpenHashMap())

    private val frames = arrayListOf(currentFrame)

    override var currentFrame = currentFrame
        private set

    override val currentScope: UInt
        get() = frames.size.toUInt()

    inline fun findNonNullValue(predicate: (Long2ObjectOpenHashMap<V>) -> V?): V? {
        frames.forEach { frame ->
            predicate(frame)?.let { return it }
        }
        return null
    }

    override fun push() {
        currentFrame = Long2ObjectOpenHashMap()
        frames += currentFrame
    }

    override fun pop(n: UInt) {
        repeat(n.toInt()) { frames.removeLast() }
        currentFrame = frames.last()
    }
}

internal class ScopedLinkedFrame<T> private constructor(
    private var current: LinkedFrame<T>,
    private val createNewFrame: () -> T,
    private val copyFrame: (T) -> T
) : ScopedFrame<T> {
    constructor(
        currentFrame: T,
        createNewFrame: () -> T,
        copyFrame: (T) -> T
    ) : this(LinkedFrame(currentFrame), createNewFrame, copyFrame)

    constructor(
        createNewFrame: () -> T,
        copyFrame: (T) -> T
    ) : this(createNewFrame(), createNewFrame, copyFrame)

    override val currentFrame: T
        get() = current.value

    override val currentScope: UInt
        get() = current.scope

    fun stacked(): ArrayDeque<T> = ArrayDeque<T>().also { stack ->
        forEachReversed { frame ->
            stack.addLast(frame)
        }
    }

    inline fun <V> findNonNullValue(predicate: (T) -> V?): V? {
        forEachReversed { frame ->
            predicate(frame)?.let { return it }
        }
        return null
    }

    override fun push() {
        current = LinkedFrame(createNewFrame(), current)
    }

    override fun pop(n: UInt) {
        current = current.previous ?: throw IllegalStateException("Can't pop the bottom scope")
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
