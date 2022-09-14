package org.ksmt.solver.z3

import com.microsoft.z3.FuncDecl
import com.microsoft.z3.Model
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KSort

open class KZ3Model(
    private val model: Model,
    private val ctx: KContext,
    private val internCtx: KZ3InternalizationContext,
    private val internalizer: KZ3ExprInternalizer,
    private val converter: KZ3ExprConverter
) : KModel {
    override val declarations: Set<KDecl<*>> by lazy {
        with(converter) {
            model.decls.mapTo(hashSetOf()) { it.convert<KSort>() }
        }
    }

    private val interpretations = hashMapOf<KDecl<*>, KModel.KFuncInterp<*>?>()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ensureContextActive()

        val z3Expr = with(internalizer) { expr.internalize() }
        val z3Result = model.eval(z3Expr, isComplete)

        return with(converter) { z3Result.convert() }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? =
        interpretations.getOrPut(decl) {
            ensureContextActive()

            if (decl !in declarations) return@getOrPut null

            val z3Decl = with(internalizer) { decl.internalizeDecl() }

            when (z3Decl) {
                in model.constDecls -> constInterp<T>(z3Decl)
                in model.funcDecls -> funcInterp<T>(z3Decl)
                else -> error("decl $decl is in model declarations but not present in model")
            }
        } as? KModel.KFuncInterp<T>

    private fun <T : KSort> constInterp(decl: FuncDecl<*>): KModel.KFuncInterp<T>? {
        val z3Expr = model.getConstInterp(decl) ?: return null

        val expr = with(converter) { z3Expr.convert<T>() }

        return with(ctx) {
            KModel.KFuncInterp(sort = expr.sort, vars = emptyList(), entries = emptyList(), default = expr)
        }
    }

    private fun <T : KSort> funcInterp(decl: FuncDecl<*>): KModel.KFuncInterp<T>? = with(converter) {
        val z3Interp = model.getFuncInterp(decl) ?: return null

        val varSorts = decl.domain.map { it.convert<KSort>() }
        val vars = varSorts.map { with(ctx) { it.mkFreshConst("x") } }
        val z3Vars = vars.map { with(internalizer) { it.internalize() } }.toTypedArray()

        val entries = z3Interp.entries.map { entry ->
            val args = entry.args.map { it.substituteVars(z3Vars).convert<KSort>() }
            val value = entry.value.substituteVars(z3Vars).convert<T>()
            KModel.KFuncInterpEntry(args, value)
        }

        val default = z3Interp.getElse().substituteVars(z3Vars).convert<T>()
        val sort = decl.range.convert<T>()
        val varDecls = vars.map { with(ctx) { it.decl } }

        return KModel.KFuncInterp(sort, varDecls, entries, default)
    }

    override fun detach(): KModel {
        val interpretations = declarations.associateWith {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        return KModelImpl(ctx, interpretations)
    }

    private fun ensureContextActive() = check(internCtx.isActive) { "Context already closed" }
}
