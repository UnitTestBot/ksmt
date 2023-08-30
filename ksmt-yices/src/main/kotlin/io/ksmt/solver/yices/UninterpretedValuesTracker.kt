package io.ksmt.solver.yices

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap

class UninterpretedValuesTracker internal constructor(
    private val ctx: KContext,
    private val scopedExpressions: ScopedFrame<HashSet<KExpr<*>>>,
    private val uninterpretedValues: ScopedFrame<HashMap<KUninterpretedSort, HashSet<KUninterpretedSortValue>>>,
    private val expressionLevels: Object2IntOpenHashMap<KExpr<*>>
) {
    private var analyzer: ExprUninterpretedValuesAnalyzer = createNewAnalyzer()

    private fun createNewAnalyzer() = ExprUninterpretedValuesAnalyzer(
        ctx,
        scopedExpressions,
        uninterpretedValues,
        expressionLevels
    )

    fun expressionUse(expr: KExpr<*>) {
        if (expr in scopedExpressions.currentFrame) return
        analyzer.apply(expr)
    }

    fun expressionSave(expr: KExpr<*>) {
        if (scopedExpressions.currentFrame.add(expr)) {
            expressionLevels.put(expr, scopedExpressions.currentScope.toInt())
        }
    }

    fun addToCurrentLevel(value: KUninterpretedSortValue) {
        analyzer.addToCurrentLevel(value)
    }

    fun getUninterpretedSortValues(sort: KUninterpretedSort) = hashSetOf<KUninterpretedSortValue>().apply {
        uninterpretedValues.forEach { frame ->
            frame[sort]?.also { this += it }
        }
    }

    fun push() {
        scopedExpressions.push()
        uninterpretedValues.push()
    }

    fun pop(n: UInt) {
        scopedExpressions.pop(n)
        uninterpretedValues.pop(n)

        analyzer = createNewAnalyzer()
    }

    private class ExprUninterpretedValuesAnalyzer(
        ctx: KContext,
        val scopedExpressions: ScopedFrame<HashSet<KExpr<*>>>,
        val uninterpretedValues: ScopedFrame<HashMap<KUninterpretedSort, HashSet<KUninterpretedSortValue>>>,
        val expressionLevels: Object2IntOpenHashMap<KExpr<*>>
    ) : KNonRecursiveTransformer(ctx) {

        fun addToCurrentLevel(value: KUninterpretedSortValue) {
            uninterpretedValues.currentFrame.getOrPut(value.sort) { hashSetOf() } += value
        }

        override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
            if (scopedExpressions.currentFrame.add(expr))
                expressionLevels[expr] = scopedExpressions.currentScope.toInt()
            return super.transformExpr(expr)
        }

        override fun transform(expr: KUninterpretedSortValue): KExpr<KUninterpretedSort> {
            addToCurrentLevel(expr)
            return super.transform(expr)
        }

        override fun <T : KSort> exprTransformationRequired(expr: KExpr<T>): Boolean {
            val frameLevel = expressionLevels.getInt(expr)
            if (frameLevel < scopedExpressions.currentScope.toInt()) {
                // If expr is valid on its level we don't need to move it
                return expr !in scopedExpressions.getFrame(frameLevel)
            }
            return super.exprTransformationRequired(expr)
        }
    }
}
