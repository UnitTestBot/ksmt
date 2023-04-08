package org.ksmt.solver.z3

import com.microsoft.z3.Model
import com.microsoft.z3.Native
import com.microsoft.z3.evalNative
import com.microsoft.z3.getConstInterp
import com.microsoft.z3.getFuncInterp
import com.microsoft.z3.getNativeConstDecls
import com.microsoft.z3.getNativeFuncDecls
import com.microsoft.z3.getNativeSorts
import com.microsoft.z3.getSortUniverse
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.mkFreshConst
import org.ksmt.utils.uncheckedCast

open class KZ3Model(
    private val model: Model,
    private val ctx: KContext,
    private val z3Ctx: KZ3Context,
    private val internalizer: KZ3ExprInternalizer,
    private val converter: KZ3ExprConverter
) : KModel {
    private val constantDeclarations: Set<KDecl<*>> by lazy {
        model.getNativeConstDecls().convertToSet { it.convertDecl<KSort>() }
    }

    private val functionDeclarations: Set<KDecl<*>> by lazy {
        model.getNativeFuncDecls().convertToSet { it.convertDecl<KSort>() }
    }

    override val uninterpretedSorts: Set<KUninterpretedSort> by lazy {
        model.getNativeSorts().convertToSet { it.convertSort() as KUninterpretedSort }
    }

    override val declarations: Set<KDecl<*>> by lazy {
        constantDeclarations + functionDeclarations
    }

    private val interpretations = hashMapOf<KDecl<*>, KModel.KFuncInterp<*>?>()
    private val uninterpretedSortsUniverses = hashMapOf<KUninterpretedSort, Set<KExpr<KUninterpretedSort>>>()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)
        ensureContextActive()

        val z3Expr = with(internalizer) { expr.internalizeExpr() }
        val z3Result = model.evalNative(z3Expr, isComplete)

        return with(converter) { z3Result.convertExpr() }
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? =
        interpretations.getOrPut(decl) {
            ctx.ensureContextMatch(decl)
            ensureContextActive()

            if (decl !in declarations) return@getOrPut null

            val z3Decl = with(internalizer) { decl.internalizeDecl() }

            when (decl) {
                in constantDeclarations -> constInterp(decl, z3Decl)
                in functionDeclarations -> funcInterp(decl, z3Decl)
                else -> error("decl $decl is in model declarations but not present in model")
            }
        }?.uncheckedCast()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KExpr<KUninterpretedSort>>? =
        uninterpretedSortsUniverses.getOrPut(sort) {
            ctx.ensureContextMatch(sort)

            if (sort !in uninterpretedSorts) {
                return@getOrPut emptySet()
            }

            val z3Sort = with(internalizer) { sort.internalizeSort() }
            val z3SortUniverse = model.getSortUniverse(z3Sort)

            z3SortUniverse.convertToSet { it.convertExpr() }
        }

    private fun <T : KSort> constInterp(decl: KDecl<T>, z3Decl: Long): KModel.KFuncInterp<T>? {
        val z3Interp = model.getConstInterp(z3Decl) ?: return null
        val expr = with(converter) { z3Interp.convertExpr<T>() }
        return KModel.KFuncInterp(decl = decl, vars = emptyList(), entries = emptyList(), default = expr)
    }

    private fun <T : KSort> funcInterp(decl: KDecl<T>, z3Decl: Long): KModel.KFuncInterp<T>? = with(converter) {
        val z3Interp = model.getFuncInterp(z3Decl) ?: return null

        val vars = decl.argSorts.map { it.mkFreshConst("x") }
        val z3Vars = LongArray(vars.size) {
            with(internalizer) { vars[it].internalizeExpr() }
        }

        val entries = z3Interp.entries.map { entry ->
            val args = entry.args.map { it.substituteVarsAndConvert<KSort>(z3Vars) }
            val value = entry.value.substituteVarsAndConvert<T>(z3Vars)
            KModel.KFuncInterpEntry(args, value)
        }

        val default = z3Interp.elseExpr.substituteVarsAndConvert<T>(z3Vars)
        val varDecls = vars.map { it.decl }

        return KModel.KFuncInterp(decl, varDecls, entries, default)
    }

    override fun detach(): KModel {
        val interpretations = declarations.associateWith {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        return KModelImpl(ctx, interpretations, uninterpretedSortsUniverses)
    }

    private fun ensureContextActive() = check(z3Ctx.isActive) { "Context already closed" }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }

    private fun <T> LongArray.convertToSet(convert: KZ3ExprConverter.(Long) -> T): Set<T> =
        mapTo(hashSetOf()) {
            converter.convert(it)
        }

    private fun <T : KSort> Long.substituteVarsAndConvert(vars: LongArray): KExpr<T> {
        val preparedExpr = z3Ctx.temporaryAst(
            Native.substituteVars(z3Ctx.nCtx, this, vars.size, vars)
        )
        val convertedExpr = with(converter) {
            preparedExpr.convertExpr<T>()
        }
        z3Ctx.releaseTemporaryAst(preparedExpr)
        return convertedExpr
    }
}
