package io.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.github.cvc5.Sort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KSortVisitor
import io.ksmt.sort.KUninterpretedSort


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

    override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>): Sort =
        cvc5Ctx.internalizeSort(sort) {
            val d0 = sort.domain0.internalizeCvc5Sort()
            val d1 = sort.domain1.internalizeCvc5Sort()

            val domain = nSolver.mkTupleSort(arrayOf(d0, d1))
            val range = sort.range.internalizeCvc5Sort()
            nSolver.mkArraySort(domain, range)
        }

    override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>): Sort =
        cvc5Ctx.internalizeSort(sort) {
            val d0 = sort.domain0.internalizeCvc5Sort()
            val d1 = sort.domain1.internalizeCvc5Sort()
            val d2 = sort.domain2.internalizeCvc5Sort()

            val domain = nSolver.mkTupleSort(arrayOf(d0, d1, d2))
            val range = sort.range.internalizeCvc5Sort()
            nSolver.mkArraySort(domain, range)
        }

    override fun <R : KSort> visit(sort: KArrayNSort<R>): Sort =
        cvc5Ctx.internalizeSort(sort) {
            val domainSorts = sort.domainSorts.map { it.internalizeCvc5Sort() }
            val domain = nSolver.mkTupleSort(domainSorts.toTypedArray())
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
        // uninterpreted sorts incremental collection optimization
        cvc5Ctx.addUninterpretedSort(sort)
        nSolver.mkUninterpretedSort(sort.name)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> T.internalizeCvc5Sort() = accept(this@KCvc5SortInternalizer)
}
