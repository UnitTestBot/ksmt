package org.ksmt.solver.cvc5

import io.github.cvc5.Term
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.mkFreshConst

open class KCvc5Model(
    private val ctx: KContext,
    private val cvc5Ctx: KCvc5Context,
    private val converter: KCvc5ExprConverter,
    private val internalizer: KCvc5ExprInternalizer,
    override val declarations: Set<KDecl<*>>,
    override val uninterpretedSorts: Set<KUninterpretedSort>
) : KModel {

    private val interpretations = hashMapOf<KDecl<*>, KModel.KFuncInterp<*>?>()
    private val uninterpretedSortsUniverses = hashMapOf<KUninterpretedSort, Set<KExpr<KUninterpretedSort>>>()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)
        ensureContextActive()

       return KModelEvaluator(ctx, this, isComplete).apply(expr)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? = interpretations.getOrPut(decl) {
        ctx.ensureContextMatch(decl)
        ensureContextActive()

        if (decl !in declarations) return@getOrPut null

        val cvc5Decl = with(internalizer) { decl.internalizeDecl() }

        when (decl) {
            is KConstDecl<T> -> constInterp(decl, cvc5Decl)
            is KFuncDecl<T> -> funcInterp(decl, cvc5Decl)
            else -> error("Unknown declaration")
        }
    } as? KModel.KFuncInterp<T>


    // cvc5 function interpretation - declaration is Term of kind Lambda
    private fun <T : KSort> funcInterp(
        decl: KDecl<T>,
        internalizedDecl: Term
    ): KModel.KFuncInterp<T> = with(converter) {
        val cvc5Interp = cvc5Ctx.nativeSolver.getValue(internalizedDecl)

        val vars = decl.argSorts.map { it.mkFreshConst("x") }
        val cvc5Vars = vars.map { with(internalizer) { it.internalizeWithNoAxiomsAllowed() } }.toTypedArray()

        val cvc5InterpArgs = cvc5Interp.getChild(0).getChildren()
        val cvc5FreshVarsInterp = cvc5Interp.substitute(cvc5InterpArgs, cvc5Vars)

        val defaultBody = cvc5FreshVarsInterp.getChild(1).convertExpr<T>()

        KModel.KFuncInterp(decl, vars.map { it.decl }, emptyList(), defaultBody)
    }

    private fun <T : KSort> constInterp(decl: KDecl<T>, const: Term): KModel.KFuncInterp<T> = with(converter) {
        val cvc5Interp = cvc5Ctx.nativeSolver.getValue(const)
        val interp = cvc5Interp.convertExpr<T>()

        KModel.KFuncInterp(decl = decl, vars = emptyList(), entries = emptyList(), default = interp)
    }

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KExpr<KUninterpretedSort>>? =
        uninterpretedSortsUniverses.getOrPut(sort) {
            ctx.ensureContextMatch(sort)
            ensureContextActive()

            if (sort !in uninterpretedSorts) {
                return@getOrPut emptySet()
            }

            val cvc5Sort = with(internalizer) { sort.internalizeSort() }
            val cvc5SortUniverse = cvc5Ctx.nativeSolver.getModelDomainElements(cvc5Sort)

            with(converter) { cvc5SortUniverse.mapTo(hashSetOf()) { it.convertExpr() } }
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

    private fun ensureContextActive() = check(cvc5Ctx.isActive) { "Context already closed" }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }
}