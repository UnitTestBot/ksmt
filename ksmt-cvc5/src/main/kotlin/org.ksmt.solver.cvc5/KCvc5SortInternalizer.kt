package org.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.github.cvc5.Sort
import org.ksmt.sort.*


open class KCvc5SortInternalizer(
    private val cvc5Ctx: KCvc5Context
) : KSortVisitor<Sort> {
    private val nSolver: Solver = cvc5Ctx.nativeSolver

    override fun visit(sort: KBoolSort): Sort = cvc5Ctx.internalizeSort(sort) {
        nSolver.booleanSort
    }

    override fun visit(sort: KIntSort): Sort = cvc5Ctx.internalizeSort(sort) {
        nSolver.integerSort
    }

    override fun visit(sort: KRealSort): Sort = cvc5Ctx.internalizeSort(sort) {
        nSolver.realSort
    }

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): Sort =
        cvc5Ctx.internalizeSort(sort) {
            val domain = sort.domain.internalizeCvc5Sort()
            val range = sort.range.internalizeCvc5Sort()
            nSolver.mkArraySort(domain, range)
        }

    override fun visit(sort: KFpRoundingModeSort): Sort = cvc5Ctx.internalizeSort(sort) {
        nSolver.roundingModeSort
    }

    override fun <T : KBvSort> visit(sort: T): Sort = cvc5Ctx.internalizeSort(sort) {
        val size = sort.sizeBits.toInt()
        nSolver.mkBitVectorSort(size)
    }

    override fun <S : KFpSort> visit(sort: S): Sort = cvc5Ctx.internalizeSort(sort) {
        val exp = sort.exponentBits.toInt()
        val significand = sort.significandBits.toInt()
        nSolver.mkFloatingPointSort(exp, significand)
    }

    override fun visit(sort: KUninterpretedSort): Sort = cvc5Ctx.internalizeSort(sort) {
        nSolver.mkUninterpretedSort(sort.name)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.internalizeCvc5Sort() = accept(this@KCvc5SortInternalizer)
}
