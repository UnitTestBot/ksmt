package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.sort.KSort

open class KModelImpl(
    val ctx: KContext,
    private val interpretations: Map<KDecl<*>, KModel.KFuncInterp<*>>
) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = interpretations.keys

    override fun <T : KSort> eval(
        expr: KExpr<T>,
        isComplete: Boolean
    ): KExpr<T> {
        val evaluator = KModelEvaluator(ctx, this, isComplete)
        return evaluator.apply(expr)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(
        decl: KDecl<T>
    ): KModel.KFuncInterp<T>? = interpretations[decl] as? KModel.KFuncInterp<T>

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
