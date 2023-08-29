package io.ksmt.utils

object Permutations {

    fun <T> getPermutations(set: Set<T>): Set<List<T>> {
        fun <T> allPermutations(list: List<T>): Set<List<T>> {
            val result: MutableSet<List<T>> = mutableSetOf()
            if (list.isEmpty())
                result.add(list)
            for (item in list) {
                allPermutations(list - item).forEach { tail ->
                    result.add(tail + item)
                }
            }
            return result
        }

        if (set.isEmpty()) return setOf(emptyList())
        return allPermutations(set.toList())
    }
}