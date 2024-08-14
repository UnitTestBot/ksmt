package io.ksmt.solver.cvc5

import io.github.cvc5.Sort
import io.github.cvc5.Term
import it.unimi.dsi.fastutil.Hash
import it.unimi.dsi.fastutil.HashCommon
import it.unimi.dsi.fastutil.objects.Object2ObjectOpenCustomHashMap
import java.util.TreeMap

class KCvc5TermMap<V>(
    private val data: MutableMap<Term, V> = Object2ObjectOpenCustomHashMap(KCvc5TermHash)
) : MutableMap<Term, V> by data

class KCvc5SortMap<V>(
    private val data: MutableMap<Sort, V> = TreeMap<Sort, V>()
) : MutableMap<Sort, V> by data

private object KCvc5TermHash : Hash.Strategy<Term> {
    override fun equals(p0: Term?, p1: Term?): Boolean =
        p0 == p1

    override fun hashCode(p0: Term?): Int {
        if (p0 == null) return 0
        return HashCommon.long2int(p0.id)
    }
}
