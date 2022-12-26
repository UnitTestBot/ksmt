package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.solver.KModel
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.model.DefaultValueSampler.Companion.sampleValue
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.mkFreshConst
import org.ksmt.utils.mkFreshConstDecl

open class KBitwuzlaModel(
    private val ctx: KContext,
    private val bitwuzlaCtx: KBitwuzlaContext,
    private val internalizer: KBitwuzlaExprInternalizer,
    private val converter: KBitwuzlaExprConverter
) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = bitwuzlaCtx.declaredConstants()

    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> = bitwuzlaCtx.bitwuzlaTry {
        with(ctx) {
            ctx.ensureContextMatch(expr)
            bitwuzlaCtx.ensureActive()

            val term = with(internalizer) { expr.internalize() }
            val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)

            with(converter) {
                val convertedExpr = value.convertExpr(expr.sort)

                if (!isComplete) return convertedExpr

                convertedExpr.accept(ModelCompleter(expr, incompleteDecls))
            }
        }
    }

    // Uninterpreted sorts are not supported in bitwuzla
    override val uninterpretedSorts: Set<KUninterpretedSort>
        get() = emptySet()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KExpr<KUninterpretedSort>>? = null

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(
        decl: KDecl<T>
    ): KModel.KFuncInterp<T> = with(ctx) {
        ensureContextMatch(decl)

        val interpretation = interpretations.getOrPut(decl) {
            bitwuzlaCtx.bitwuzlaTry {
                bitwuzlaCtx.ensureActive()

                val term = with(internalizer) {
                    bitwuzlaCtx.mkConstant(decl, decl.bitwuzlaFunctionSort())
                }

                when {
                    Native.bitwuzlaTermIsArray(term) -> arrayInterpretation(decl, term)
                    Native.bitwuzlaTermIsFun(term) -> functionInterpretation(decl, term)
                    else -> {
                        val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
                        val convertedValue = with(converter) { value.convertExpr(decl.sort) }
                        KModel.KFuncInterp(
                            decl = decl,
                            vars = emptyList(),
                            entries = emptyList(),
                            default = convertedValue
                        )
                    }
                }
            }
        }

        interpretation as KModel.KFuncInterp<T>
    }

    private fun <T : KSort> functionInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = with(converter) {
        check(decl is KFuncDecl<T>) { "function expected but actual is $decl" }

        val entries = mutableListOf<KModel.KFuncInterpEntry<T>>()
        val interp = Native.bitwuzlaGetFunValue(bitwuzlaCtx.bitwuzla, term)

        check(interp.arity == decl.argSorts.size) {
            "function arity mismatch: ${interp.arity} and ${decl.argSorts.size}"
        }

        val vars = decl.argSorts.map { it.mkFreshConstDecl("x") }
        for (i in 0 until interp.size) {
            // Don't substitute vars since arguments in Bitwuzla model are always constants
            val args = interp.args[i].zip(decl.argSorts) { arg, sort -> arg.convertExpr(sort) }
            val value = interp.values[i].convertExpr(decl.sort)
            entries += KModel.KFuncInterpEntry(args, value)
        }

        KModel.KFuncInterp(
            decl = decl,
            vars = vars,
            entries = entries,
            default = null
        )
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : KSort> KContext.arrayInterpretation(
        decl: KDecl<T>,
        term: BitwuzlaTerm
    ): KModel.KFuncInterp<T> = with(converter) {
        val sort = decl.sort as KArraySort<*, *>
        val entries = mutableListOf<KModel.KFuncInterpEntry<KSort>>()
        val interp = Native.bitwuzlaGetArrayValue(bitwuzlaCtx.bitwuzla, term)

        for (i in 0 until interp.size) {
            val index = interp.indices[i].convertExpr(sort.domain)
            val value = interp.values[i].convertExpr(sort.range)
            entries += KModel.KFuncInterpEntry(listOf(index), value)
        }

        val default = interp.defaultValue?.convertExpr(sort.range)
        val arrayInterpDecl = mkFreshFuncDecl("array", sort.range, listOf(sort.domain))
        val arrayInterpIndexDecl = mkFreshConstDecl("idx", sort.domain)

        interpretations[arrayInterpDecl] = KModel.KFuncInterp(
            decl = arrayInterpDecl,
            vars = listOf(arrayInterpIndexDecl),
            entries = entries,
            default = default
        )

        KModel.KFuncInterp(
            decl = decl,
            vars = emptyList(),
            entries = emptyList(),
            default = mkFunctionAsArray<KSort, KSort>(arrayInterpDecl) as KExpr<T>
        )
    }

    override fun detach(): KModel {
        declarations.forEach { interpretation(it) }
        return KModelImpl(ctx, interpretations.toMutableMap(), uninterpretedSortsUniverses = emptyMap())
    }

    /**
     * Generate concrete values instead of unknown constants.
     * */
    inner class ModelCompleter(
        private val evaluatedExpr: KExpr<*>,
        private val incompleteDecls: Set<KDecl<*>>
    ) : KTransformer {
        override val ctx: KContext = this@KBitwuzlaModel.ctx

        override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = with(ctx) {
            if (expr == evaluatedExpr) {
                return expr.sort.sampleValue()
            }

            return super.transformExpr(expr)
        }

        override fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = with(ctx) {
            if (expr.decl in incompleteDecls) {
                return expr.sort.sampleValue()
            }

            return super.transformApp(expr)
        }
    }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }
}
