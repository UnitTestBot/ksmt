package org.ksmt.solver.z3

import com.microsoft.z3.Context
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
    private val z3Ctx: Context,
    private val converter: KZ3ExprConverter
) : KModel {
    override val declarations: Set<KDecl<*>> by lazy {
        with(converter) {
            model.decls.mapTo(mutableSetOf()) { it.convert<KSort>() }
        }
    }

    override fun <T : KSort> eval(expr: KExpr<T>, complete: Boolean): KExpr<T> {
        ensureContextActive()
        val z3Expr = with(internalizer) { expr.internalize() }
        val z3Result = model.eval(z3Expr, complete)
        return with(converter) { z3Result.convert() }
    }

    @Suppress("ReturnCount")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? {
        ensureContextActive()
        if (decl !in declarations) return null
        val z3Decl = with(internalizer) { decl.internalize() }
        if (z3Decl in model.constDecls) return constInterp(z3Decl)
        return funcInterp(z3Decl)
    }

    @Suppress("MemberVisibilityCanBePrivate")
    fun <T : KSort> constInterp(decl: FuncDecl): KModel.KFuncInterp<T>? {
        val z3Expr = model.getConstInterp(decl) ?: return null
        val expr = with(converter) { z3Expr.convert<T>() }
        return with(ctx) {
            KModel.KFuncInterp(sort = expr.sort, vars = emptyList(), entries = emptyList(), default = expr)
        }
    }

    @Suppress("MemberVisibilityCanBePrivate", "ForbiddenComment")
    fun <T : KSort> funcInterp(decl: FuncDecl): KModel.KFuncInterp<T>? = with(converter) {
        val z3Interp = model.getFuncInterp(decl) ?: return null
        // todo: convert z3 vars or generate fresh const by ourself
        val z3Vars = decl.domain.map { z3Ctx.mkFreshConst("x", it) }.toTypedArray()
        val vars = z3Vars.map { it.funcDecl.convert<KSort>() }
        val entries = z3Interp.entries.map { entry ->
            val args = entry.args.map { it.substituteVars(z3Vars).convert<KSort>() }
            val value = entry.value.substituteVars(z3Vars).convert<T>()
            KModel.KFuncInterpEntry(args, value)
        }
        val default = z3Interp.getElse().substituteVars(z3Vars).convert<T>()
        val sort = decl.range.convert<T>()
        return KModel.KFuncInterp(sort, vars, entries, default)
    }

    override fun detach(): KModel {
        val interpretations = declarations.associateWith {
            interpretation(it) ?: error("missed interpretation for $it")
        }
        return KModelImpl(ctx, interpretations)
    }

    private fun ensureContextActive() = check(internCtx.isActive) { "Context already closed" }
}
