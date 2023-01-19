package org.ksmt.cache

interface KInternedObject {
    fun internEquals(other: Any): Boolean
    fun internHashCode(): Int
}
