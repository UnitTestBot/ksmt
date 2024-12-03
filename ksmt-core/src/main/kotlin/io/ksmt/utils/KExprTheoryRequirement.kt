package io.ksmt.utils

import io.ksmt.KContext
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KExistentialQuantifier
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionApp
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KUniversalQuantifier
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.KTheory.UF
import io.ksmt.sort.*

class KExprTheoryRequirement(ctx: KContext) : KNonRecursiveTransformer(ctx) {
    val usedTheories = hashSetOf<KTheory>()

    var hasQuantifiers: Boolean = false
        private set

    private val sortRequirementCollector = Sort2TheoryRequirement()

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
        expr.sort.accept(sortRequirementCollector)
        return super.transformExpr(expr)
    }

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> {
        if (expr.args.isNotEmpty()) {
            usedTheories += UF
        }
        return super.transform(expr)
    }

    override fun transform(expr: KExistentialQuantifier): KExpr<KBoolSort> {
        hasQuantifiers = true
        return super.transform(expr)
    }

    override fun transform(expr: KUniversalQuantifier): KExpr<KBoolSort> {
        hasQuantifiers = true
        return super.transform(expr)
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> {
        usedTheories += if (expr.sort is KIntSort) NIA else NRA
        return super.transform(expr)
    }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> {
        usedTheories += if (expr.sort is KIntSort) NIA else NRA
        return super.transform(expr)
    }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> {
        usedTheories += if (expr.sort is KIntSort) NIA else NRA
        return super.transform(expr)
    }

    private inner class Sort2TheoryRequirement : KSortVisitor<Unit> {
        override fun visit(sort: KBoolSort) {
        }

        override fun visit(sort: KIntSort) {
            usedTheories += LIA
        }

        override fun visit(sort: KRealSort) {
            usedTheories += LRA
        }

        override fun visit(sort: KStringSort) {
            TODO("Not yet implemented")
        }

        override fun visit(sort: KRegexSort) {
            TODO("Not yet implemented")
        }

        override fun <S : KBvSort> visit(sort: S) {
            usedTheories += BV
        }

        override fun <S : KFpSort> visit(sort: S) {
            usedTheories += FP
        }

        override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>) {
            usedTheories += Array
            sort.range.accept(this)
            sort.domainSorts.forEach { it.accept(this) }
        }

        override fun <D0 : KSort, D1 : KSort, R : KSort> visit(sort: KArray2Sort<D0, D1, R>) {
            usedTheories += Array
            sort.range.accept(this)
            sort.domainSorts.forEach { it.accept(this) }
        }

        override fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(sort: KArray3Sort<D0, D1, D2, R>) {
            usedTheories += Array
            sort.range.accept(this)
            sort.domainSorts.forEach { it.accept(this) }
        }

        override fun <R : KSort> visit(sort: KArrayNSort<R>) {
            usedTheories += Array
            sort.range.accept(this)
            sort.domainSorts.forEach { it.accept(this) }
        }

        override fun visit(sort: KFpRoundingModeSort) {
            usedTheories += FP
        }

        override fun visit(sort: KUninterpretedSort) {
            usedTheories += UF
        }
    }
}
