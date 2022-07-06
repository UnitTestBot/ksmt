package org.ksmt.cache

import java.lang.ref.WeakReference
import java.util.WeakHashMap

class Cache0<T>(val builder: () -> T) {
    private var value: Any? = UNINITIALIZED

    @Suppress("UNCHECKED_CAST")
    fun create(): T {
        if (UNINITIALIZED === value) {
            value = builder()
        }
        return value as T
    }

    companion object {
        private val UNINITIALIZED = Any()
    }
}

class Cache1<T, A0>(val builder: (A0) -> T) {
    private val cache = WeakHashMap<A0, WeakReference<T>>()
    fun create(a0: A0): T {
        val currentNode = cache[a0]?.get()
        if (currentNode != null) return currentNode
        val newNode = builder(a0)
        cache[a0] = WeakReference(newNode)
        return newNode
    }
}

class Cache2<T, A0, A1>(val builder: (A0, A1) -> T) {
    private val cache = WeakHashMap<A0, WeakHashMap<A1, WeakReference<T>>>()
    fun create(a0: A0, a1: A1): T {
        val node0 = cache.getOrPut(a0) { WeakHashMap() }
        val currentNode = node0[a1]?.get()
        if (currentNode != null) return currentNode
        val newNode = builder(a0, a1)
        node0[a1] = WeakReference(newNode)
        return newNode
    }
}

class Cache3<T, A0, A1, A2>(val builder: (A0, A1, A2) -> T) {
    private val cache = WeakHashMap<A0, WeakHashMap<A1, WeakHashMap<A2, WeakReference<T>>>>()
    fun create(a0: A0, a1: A1, a2: A2): T {
        val node0 = cache.getOrPut(a0) { WeakHashMap() }
        val node1 = node0.getOrPut(a1) { WeakHashMap() }
        val currentNode = node1[a2]?.get()
        if (currentNode != null) return currentNode
        val newNode = builder(a0, a1, a2)
        node1[a2] = WeakReference(newNode)
        return newNode
    }
}

class Cache4<T, A0, A1, A2, A3>(val builder: (A0, A1, A2, A3) -> T) {
    private val cache = WeakHashMap<A0, WeakHashMap<A1, WeakHashMap<A2, WeakHashMap<A3, WeakReference<T>>>>>()
    @Suppress("unused")
    fun create(a0: A0, a1: A1, a2: A2, a3: A3): T {
        val node0 = cache.getOrPut(a0) { WeakHashMap() }
        val node1 = node0.getOrPut(a1) { WeakHashMap() }
        val node2 = node1.getOrPut(a2) { WeakHashMap() }
        val currentNode = node2[a3]?.get()
        if (currentNode != null) return currentNode
        val newNode = builder(a0, a1, a2, a3)
        node2[a3] = WeakReference(newNode)
        return newNode
    }
}

fun <T> mkCache(builder: () -> T) = Cache0(builder)
fun <T, A0> mkCache(builder: (A0) -> T) = Cache1(builder)
fun <T, A0, A1> mkCache(builder: (A0, A1) -> T) = Cache2(builder)
fun <T, A0, A1, A2> mkCache(builder: (A0, A1, A2) -> T) = Cache3(builder)
@Suppress("unused")
fun <T, A0, A1, A2, A3> mkCache(builder: (A0, A1, A2, A3) -> T) = Cache4(builder)
