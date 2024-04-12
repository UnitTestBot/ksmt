package io.ksmt.solver.cvc5

import io.github.cvc5.AbstractPointer
import io.github.cvc5.Kind
import io.github.cvc5.Op
import io.github.cvc5.Sort
import io.github.cvc5.Term
import io.github.cvc5.TermManager
import it.unimi.dsi.fastutil.Hash
import it.unimi.dsi.fastutil.HashCommon
import it.unimi.dsi.fastutil.objects.ObjectOpenCustomHashSet

class KCvc5TermManager : AutoCloseable {
    val termManager: TermManager = TermManager()

    fun mkTerm(kind: Kind): Term =
        termManager.mkTerm(kind).also { registerPointer(it) }

    fun mkTerm(kind: Kind, arg: Term): Term =
        termManager.mkTerm(kind, arg).also { registerPointer(it) }

    fun mkTerm(kind: Kind, arg0: Term, arg1: Term): Term =
        termManager.mkTerm(kind, arg0, arg1).also { registerPointer(it) }

    fun mkTerm(kind: Kind, arg0: Term, arg1: Term, arg2: Term): Term =
        termManager.mkTerm(kind, arg0, arg1, arg2).also { registerPointer(it) }

    fun mkTerm(kind: Kind, args: Array<Term>): Term =
        termManager.mkTerm(kind, args).also { registerPointer(it) }

    fun mkTerm(op: Op): Term =
        termManager.mkTerm(op).also { registerPointer(it) }

    fun mkTerm(op: Op, arg0: Term): Term =
        termManager.mkTerm(op, arg0).also { registerPointer(it) }

    fun mkTerm(op: Op, arg1: Term, arg2: Term): Term =
        termManager.mkTerm(op, arg1, arg2).also { registerPointer(it) }

    fun mkTerm(op: Op, arg0: Term, arg1: Term, arg2: Term): Term =
        termManager.mkTerm(op, arg0, arg1, arg2).also { registerPointer(it) }

    fun mkTerm(op: Op, args: Array<Term>): Term =
        termManager.mkTerm(op, args).also { registerPointer(it) }

    fun mkQuantifier(isUniversal: Boolean, boundVars: Array<Term>, body: Term): Term {
        val kind = if (isUniversal) Kind.FORALL else Kind.EXISTS
        val quantifiedVars = mkTerm(Kind.VARIABLE_LIST, boundVars)
        return mkTerm(kind, quantifiedVars, body)
    }

    fun mkLambda(boundVars: Array<Term>, body: Term): Term {
        val lambdaVars = mkTerm(Kind.VARIABLE_LIST, boundVars)
        return mkTerm(Kind.LAMBDA, lambdaVars, body)
    }

    fun mkOp(kind: Kind): Op =
        termManager.mkOp(kind).also { registerPointer(it) }

    fun mkOp(kind: Kind, arg: String): Op =
        termManager.mkOp(kind, arg).also { registerPointer(it) }

    fun mkOp(kind: Kind, arg: Int): Op =
        termManager.mkOp(kind, arg).also { registerPointer(it) }

    fun mkOp(kind: Kind, arg0: Int, arg1: Int): Op =
        termManager.mkOp(kind, arg0, arg1).also { registerPointer(it) }

    fun mkOp(kind: Kind, args: IntArray): Op =
        termManager.mkOp(kind, args).also { registerPointer(it) }

    inline fun <reified T : AbstractPointer> builder(body: TermManager.() -> T): T =
        body(termManager).also { registerPointer(it) }

    fun termSort(term: Term): Sort = term.sort.also { registerPointer(it) }

    fun termChild(term: Term, idx: Int): Term = term.getChild(idx).also { registerPointer(it) }

    inline fun <reified T : AbstractPointer> termOp(term: Term, body: Term.() -> T): T =
        body(term).also { registerPointer(it) }

    inline fun <reified T : AbstractPointer> sortOp(sort: Sort, body: Sort.() -> T): T =
        body(sort).also { registerPointer(it) }

    private val pointers = ObjectOpenCustomHashSet(PointerHash)

    fun <T : AbstractPointer> registerPointer(ptr: T): T {
        pointers.add(ptr)
        return ptr
    }

    override fun close() {
        for (ptr in pointers) {
            ptr.deletePointer()
        }
        termManager.deletePointer()
    }

    private object PointerHash : Hash.Strategy<AbstractPointer> {
        override fun equals(p0: AbstractPointer?, p1: AbstractPointer?): Boolean {
            if (p0 == null || p1 == null) return p0 === p1
            return p0.pointer == p1.pointer
        }

        override fun hashCode(p0: AbstractPointer?): Int {
            if (p0 == null) return 0
            return HashCommon.long2int(p0.pointer)
        }
    }
}
