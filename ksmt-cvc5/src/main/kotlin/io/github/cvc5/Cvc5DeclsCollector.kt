package io.github.cvc5

import java.util.TreeSet

class Cvc5DeclsCollector(private val terms: Collection<Term>) {
    fun Term.collect(): Set<Term> = TreeSet<Term>().apply {
        when (kind) {
            Kind.CONSTANT -> add(this@collect)
            Kind.APPLY_UF -> {
                add(this@collect.getChild(0)) // 0th term is function declaration in cvc5
                for (i in 1 until this@collect.numChildren) {
                    addAll(this@collect.getChild(i).collect())
                }
            }
            else -> this@collect.asIterable().forEach { addAll(it.collect()) }
        }
    }

    fun collect(): Set<Term> = TreeSet<Term>().apply {
        terms.forEach { term ->
            this += term.collect()
        }
    }
}
