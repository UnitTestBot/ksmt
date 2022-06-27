package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.sort.KSort

open class KModelImpl(
    val ctx: KContext,
    val interpretations: Map<KDecl<*>, KModel.KFuncInterp<*>>
) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = interpretations.keys

    override fun <T : KSort> eval(expr: KExpr<T>, complete: Boolean): KExpr<T> =
        KModelEvaluator(ctx, this, complete).apply(expr)

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? =
        interpretations[decl] as? KModel.KFuncInterp<T>

    override fun detach(): KModel = this
}
