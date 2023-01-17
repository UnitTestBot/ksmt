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

    fun create(a0: A0): T = cache.computeIfAbsent(a0, builder)

    operator fun contains(key: A0): Boolean = key in cache

    override fun close() {
        cache.clear()
    }
}

class Cache2<T, A0, A1>(val builder: (A0, A1) -> T) : AutoCloseable {
    private val cache = HashMap<Pair<A0, A1>, T>()
    private val valueBuilder = { key: Pair<A0, A1> -> builder(key.first, key.second) }

    fun create(a0: A0, a1: A1): T = cache.computeIfAbsent(Pair(a0, a1), valueBuilder)

    override fun close() {
        cache.clear()
    }
}

class Cache3<T, A0, A1, A2>(val builder: (A0, A1, A2) -> T) : AutoCloseable {
    private val cache = HashMap<Triple<A0, A1, A2>, T>()
    private val valueBuilder = { key: Triple<A0, A1, A2> -> builder(key.first, key.second, key.third) }

    fun create(a0: A0, a1: A1, a2: A2): T = cache.computeIfAbsent(Triple(a0, a1, a2), valueBuilder)

    override fun close() {
        cache.clear()
    }
}

private data class Tuple4<A0, A1, A2, A3>(
    val first: A0,
    val second: A1,
    val third: A2,
    val fourth: A3
)

class Cache4<T, A0, A1, A2, A3>(val builder: (A0, A1, A2, A3) -> T) : AutoCloseable {
    private val cache = HashMap<Tuple4<A0, A1, A2, A3>, T>()
    private val valueBuilder = { key: Tuple4<A0, A1, A2, A3> ->
        builder(key.first, key.second, key.third, key.fourth)
    }

    fun create(a0: A0, a1: A1, a2: A2, a3: A3): T = cache.computeIfAbsent(Tuple4(a0, a1, a2, a3), valueBuilder)

    override fun close() {
        cache.clear()
    }
}

private data class Tuple5<A0, A1, A2, A3, A4>(
    val first: A0,
    val second: A1,
    val third: A2,
    val fourth: A3,
    val fifth: A4
)

class Cache5<T, A0, A1, A2, A3, A4>(val builder: (A0, A1, A2, A3, A4) -> T) : AutoCloseable {
    private val cache = HashMap<Tuple5<A0, A1, A2, A3, A4>, T>()
    private val valueBuilder = { key: Tuple5<A0, A1, A2, A3, A4> ->
        builder(key.first, key.second, key.third, key.fourth, key.fifth)
    }

    fun create(a0: A0, a1: A1, a2: A2, a3: A3, a4: A4): T =
        cache.computeIfAbsent(Tuple5(a0, a1, a2, a3, a4), valueBuilder)

    override fun close() {
        cache.clear()
    }
}

fun <T> mkCache(builder: () -> T) = Cache0(builder)
fun <T, A0> mkCache(builder: (A0) -> T) = Cache1(builder)
fun <T, A0, A1> mkCache(builder: (A0, A1) -> T) = Cache2(builder)
fun <T, A0, A1, A2> mkCache(builder: (A0, A1, A2) -> T) = Cache3(builder)
fun <T, A0, A1, A2, A3> mkCache(builder: (A0, A1, A2, A3) -> T) = Cache4(builder)
fun <T, A0, A1, A2, A3, A4> mkCache(builder: (A0, A1, A2, A3, A4) -> T) = Cache5(builder)
