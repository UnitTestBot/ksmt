package org.ksmt.solver.cvc5

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KUninterpretedSort

private class KUninterpretedSortCollector(private val cvc5Ctx: KCvc5Context) : KSortVisitor<Unit> {
    override fun visit(sort: KBoolSort) = Unit

    override fun visit(sort: KIntSort) = Unit

    override fun visit(sort: KRealSort) = Unit

    override fun <S : KBvSort> visit(sort: S) = Unit

    override fun <S : KFpSort> visit(sort: S) = Unit

    override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
        sort.domain.accept(this)
        sort.range.accept(this)
    }

    override fun visit(sort: KFpRoundingModeSort) = Unit

    override fun visit(sort: KUninterpretedSort) {
        cvc5Ctx.addUninterpretedSort(sort)
    }
}

internal fun KCvc5Context.collectUninterpretedSorts(decl: KDecl<*>) {
    KUninterpretedSortCollector(this).apply {
        decl.argSorts.map { it.accept(this) }
        decl.sort.accept(this)
    }
}
