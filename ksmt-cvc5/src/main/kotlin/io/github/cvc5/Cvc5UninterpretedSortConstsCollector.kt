package io.github.cvc5

import java.util.*

class Cvc5UninterpretedSortConstsCollector(private val terms: Collection<Term>) {

    fun Term.collect(sort: Sort): Set<Term> = TreeSet<Term>().apply {
        when(kind) {
            Kind.UNINTERPRETED_SORT_VALUE -> if (this@collect.sort == sort) add(this@collect)
            else -> this@collect.asIterable().forEach { addAll(it.collect(sort)) }
        }
    }

    fun collect(sort: Sort): Set<Term> = TreeSet<Term>().apply {
        terms.forEach { term ->
            this += term.collect(sort)
        }
    }
}
