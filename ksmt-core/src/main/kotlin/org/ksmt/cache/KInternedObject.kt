package org.ksmt.cache

interface KInternedObject {
    fun internEquals(other: Any): Boolean
    fun internHashCode(): Int

    companion object {
        @JvmStatic
        fun objectEquality(lhs: KInternedObject?, rhs: Any?): Boolean =
            (lhs === rhs) || (lhs !== null && rhs !== null && lhs.internEquals(rhs))
    }
}
