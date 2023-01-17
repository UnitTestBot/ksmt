package org.ksmt.cache

interface CustomObjectEquality {
    fun customEquals(other: Any): Boolean
    fun customHashCode(): Int

    companion object {
        @JvmStatic
        fun objectEquality(lhs: CustomObjectEquality?, rhs: Any?): Boolean =
            (lhs === rhs) || (lhs !== null && rhs !== null && lhs.customEquals(rhs))
    }
}
