package org.ksmt.cache

class Cache0<T>(val builder: () -> T) : AutoCloseable {
    private var value: Any? = UNINITIALIZED

    @Suppress("UNCHECKED_CAST")
    fun create(): T {
        if (UNINITIALIZED === value) {
            value = builder()
        }
        return value as T
    }

    override fun close() {
        value = UNINITIALIZED
    }

    companion object {
        private val UNINITIALIZED = Any()
    }
}

class Cache1<T, A0>(val builder: (A0) -> T) : AutoCloseable {
    private val cache = HashMap<A0, T>()
    fun create(a0: A0): T = cache.getOrPut(a0) { builder(a0) }
    override fun close() {
        cache.clear()
    }
}

class Cache2<T, A0, A1>(val builder: (A0, A1) -> T) : AutoCloseable {
    private val cache = HashMap<Pair<A0, A1>, T>()
    fun create(a0: A0, a1: A1): T = cache.getOrPut(Pair(a0, a1)) { builder(a0, a1) }
    override fun close() {
        cache.clear()
    }
}

class Cache3<T, A0, A1, A2>(val builder: (A0, A1, A2) -> T) : AutoCloseable {
    private val cache = HashMap<Triple<A0, A1, A2>, T>()
    fun create(a0: A0, a1: A1, a2: A2): T = cache.getOrPut(Triple(a0, a1, a2)) { builder(a0, a1, a2) }
    override fun close() {
        cache.clear()
    }
}

class Cache4<T, A0, A1, A2, A3>(val builder: (A0, A1, A2, A3) -> T) : AutoCloseable {
    private val cache = HashMap<List<*>, T>()

    @Suppress("unused")
    fun create(a0: A0, a1: A1, a2: A2, a3: A3): T = cache.getOrPut(listOf(a0, a1, a2, a3)) { builder(a0, a1, a2, a3) }

    override fun close() {
        cache.clear()
    }
}

fun <T> mkCache(builder: () -> T) = Cache0(builder)
fun <T, A0> mkCache(builder: (A0) -> T) = Cache1(builder)
fun <T, A0, A1> mkCache(builder: (A0, A1) -> T) = Cache2(builder)
fun <T, A0, A1, A2> mkCache(builder: (A0, A1, A2) -> T) = Cache3(builder)

@Suppress("unused")
fun <T, A0, A1, A2, A3> mkCache(builder: (A0, A1, A2, A3) -> T) = Cache4(builder)
