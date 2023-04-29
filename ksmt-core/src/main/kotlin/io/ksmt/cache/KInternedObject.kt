package io.ksmt.cache

/**
 * An object which can be interned and requires a special implementation of
 * equals and hashCode methods for the interning purposes.
 * */
interface KInternedObject {

    /**
     * [Any.equals] analogue for interning purposes.
     * */
    fun internEquals(other: Any): Boolean

    /**
     * [Any.hashCode] analogue for interning purposes.
     * */
    fun internHashCode(): Int
}
