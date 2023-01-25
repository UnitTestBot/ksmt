package io.github.cvc5

import java.util.TreeSet

class Cvc5UninterpretedSortsCollector(private val terms: Collection<Term>) {

    fun Term.collect(): Set<Sort> = TreeSet<Sort>().apply {
        when {
            sort.isUninterpretedSort -> add(sort)
            else -> this@collect.asIterable().forEach { addAll(it.collect()) }
        }
    }

    fun collect(): Set<Sort> = TreeSet<Sort>().apply {
        terms.forEach { term ->
            this += term.collect()
        }
    }
}
