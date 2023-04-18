package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.solver.KModel
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.uncheckedCast

open class KModelImpl(
    val ctx: KContext,
    private val interpretations: Map<KDecl<*>, KFuncInterp<*>>,
    private val uninterpretedSortsUniverses: Map<KUninterpretedSort, Set<KUninterpretedSortValue>>
) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = interpretations.keys

    override val uninterpretedSorts: Set<KUninterpretedSort>
        get() = uninterpretedSortsUniverses.keys

    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }

    override fun <T : KSort> eval(
        expr: KExpr<T>,
        isComplete: Boolean
    ): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        return evaluator.apply(expr)
    }

    override fun <T : KSort> interpretation(
        decl: KDecl<T>
    ): KFuncInterp<T>? {
        ctx.ensureContextMatch(decl)

        return interpretations[decl]?.uncheckedCast()
    }

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? {
        ctx.ensureContextMatch(sort)

        return uninterpretedSortsUniverses[sort]
    }

    override fun detach(): KModel = this

    override fun toString(): String = buildString {
        interpretations.forEach { (decl, interp) ->
            append(decl)
            append(":=\n\t")
            append(interp)
            appendLine()
        }
    }

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as KModelImpl

        if (ctx != other.ctx) return false
        if (interpretations != other.interpretations) return false

        return true
    }

    override fun hashCode(): Int = interpretations.hashCode()
}
