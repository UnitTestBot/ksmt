package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KSort

class KBitwuzlaModel(
    val ctx: KContext,
    val bitwuzlaCtx: KBitwuzlaContext,
    val internalizer: KBitwuzlaExprInternalizer,
    val converter: KBitwuzlaExprConverter
) : KModel {
    override val declarations: Set<KDecl<*>> by lazy {
        bitwuzlaCtx.allConstants().map {
            with(converter) { it.convert<KSort>() as KApp<*, *> }
        }.mapTo(mutableSetOf()) { with(ctx) { it.decl } }
    }

    override fun <T : KSort> eval(expr: KExpr<T>, complete: Boolean): KExpr<T> {
        bitwuzlaCtx.ensureActive()
        val term = with(internalizer) { expr.internalize() }
        val value = Native.bitwuzla_get_value(bitwuzlaCtx.bitwuzla, term)
        return with(converter) { value.convert() }
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? {

        TODO("Not yet implemented")
    }

    override fun detach(): KModel {
        TODO("Not yet implemented")
    }
}