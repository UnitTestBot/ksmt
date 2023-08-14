package io.ksmt.solver.yices

internal interface ScopedFrame<T> {
    val currentScope: UInt
    val currentFrame: T

    fun getFrame(level: Int): T

    fun any(predicate: (T) -> Boolean): Boolean

    /**
     * find value [V] in frame [T], and return it or null
     */
    fun <V> find(predicate: (T) -> V?): V?
    fun forEach(action: (T) -> Unit)

    fun push()
    fun pop(n: UInt = 1u)
}

internal class ScopedArrayFrame<T>(
    currentFrame: T,
    private inline val createNewFrame: () -> T
) : ScopedFrame<T> {
    constructor(createNewFrame: () -> T) : this(createNewFrame(), createNewFrame)

    private val frames = arrayListOf(currentFrame)

    override var currentFrame = currentFrame
        private set

    override val currentScope: UInt
        get() = frames.size.toUInt()

    override fun getFrame(level: Int): T = frames[level]

    override fun any(predicate: (T) -> Boolean): Boolean = frames.any(predicate)

    override fun <V> find(predicate: (T) -> V?): V? {
        frames.forEach { frame ->
            predicate(frame)?.let { return it }
        }
        return null
    }

    override fun forEach(action: (T) -> Unit) = frames.forEach(action)

    override fun push() {
        currentFrame = createNewFrame()
        frames += currentFrame
    }

    override fun pop(n: UInt) {
        repeat(n.toInt()) { frames.removeLast() }
        currentFrame = frames.last()
    }
}

internal class ScopedLinkedFrame<T> private constructor(
    private var current: LinkedFrame<T>,
    private inline val createNewFrame: () -> T,
    private inline val copyFrame: (T) -> T
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

    override fun getFrame(level: Int): T {
        if (level > current.scope.toInt() || level < 0) throw IllegalArgumentException("Level $level is out of scope")
        var cur: LinkedFrame<T>? = current

        while (cur != null && level < cur.scope.toInt()) {
            cur = cur.previous
        }
        return cur!!.value
    }

    override fun any(predicate: (T) -> Boolean): Boolean {
        forEachReversed { frame ->
            if (predicate(frame)) return true
        }
        return false
    }

    fun stacked(): ArrayDeque<T> = ArrayDeque<T>().also { stack ->
        forEachReversed { frame ->
            stack.addLast(frame)
        }
    }

    override fun <V> find(predicate: (T) -> V?): V? {
        forEachReversed { frame ->
            predicate(frame)?.let { return it }
        }
        return null
    }

    override fun forEach(action: (T) -> Unit) = forEachReversed(action)

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
