package io.ksmt.solver.yices

import com.sri.yices.Terms

@Suppress("SpreadOperator")
object TermUtils {
    @JvmStatic
    fun andTerm(args: YicesTermArray) = Terms.and(*args)

    @JvmStatic
    fun orTerm(args: YicesTermArray) = Terms.or(*args)

    @JvmStatic
    fun addTerm(args: YicesTermArray) = Terms.add(*args)

    @JvmStatic
    fun mulTerm(args: YicesTermArray) = Terms.mul(*args)

    @JvmStatic
    fun distinctTerm(args: YicesTermArray) = Terms.distinct(*args)

    @JvmStatic
    fun funApplicationTerm(func: YicesTerm, args: YicesTermArray) = Terms.funApplication(func, *args)
}
