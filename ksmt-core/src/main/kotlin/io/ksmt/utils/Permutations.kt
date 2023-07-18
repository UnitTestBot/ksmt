package io.ksmt.utils

object Permutations {

    fun <T> getPermutations(set: Set<T>): Set<List<T>> {
        fun <T> allPermutations(list: List<T>): Set<List<T>> {
            val result: MutableSet<List<T>> = mutableSetOf()
            for (i in list.indices) {
                allPermutations(list - list[i]).forEach { item ->
                    result.add(item + list[i])
                }
            }
            return result
        }

        if (set.isEmpty()) return setOf(emptyList())
        return allPermutations(set.toList())
    }
}