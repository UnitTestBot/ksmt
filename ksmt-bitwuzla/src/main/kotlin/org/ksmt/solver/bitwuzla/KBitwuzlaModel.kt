package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KTransformer
import org.ksmt.solver.KModel
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

open class KBitwuzlaModel(
    private val ctx: KContext,
    private val bitwuzlaCtx: KBitwuzlaContext,
    private val internalizer: KBitwuzlaExprInternalizer,
    private val converter: KBitwuzlaExprConverter
) : KModel {
    override val declarations: Set<KDecl<*>>
        get() = bitwuzlaCtx.declaredConstants()

    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>> = hashMapOf()

    override fun <T : KSort> eval(expr: KExpr<T>, complete: Boolean): KExpr<T> = with(ctx) {
        bitwuzlaCtx.ensureActive()
        val term = with(internalizer) { expr.internalize() }
        val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
        with(converter) {
            val convertedExpr = value.convertExpr(expr.sort)

            if (!complete) return convertedExpr

            val evaluatedExpr = expr
            convertedExpr.accept(object : KTransformer {
                override val ctx: KContext = this@KBitwuzlaModel.ctx
                override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
                    if (expr == evaluatedExpr) return with(valueSampler) { expr.sort.sampleValue() }
                    return super.transformExpr(expr)
                }

                override fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> {
                    if (expr.decl in incompleteDecls) return with(valueSampler) { expr.sort.sampleValue() }
                    return super.transformApp(expr)
                }
            })
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T> = interpretations.getOrPut(decl) {
        with(ctx) {
            bitwuzlaCtx.ensureActive()
            val term = with(internalizer) {
                bitwuzlaCtx.mkConstant(decl, decl.bitwuzlaFunctionSort())
            }
            when {
                Native.bitwuzlaTermIsArray(term) -> with(converter) {
                    val sort = decl.sort as KArraySort<*, *>
                    val entries = mutableListOf<KModel.KFuncInterpEntry<KSort>>()
                    val interp = Native.bitwuzlaGetArrayValue(bitwuzlaCtx.bitwuzla, term)
                    for (i in 0 until interp.size) {
                        val index = interp.indices[i].convertExpr(sort.domain)
                        val value = interp.values[i].convertExpr(sort.range)
                        entries += KModel.KFuncInterpEntry(listOf(index), value)
                    }
                    val arg = mkFreshConstDecl("var", sort.domain)
                    val default = interp.defaultValue?.convertExpr(sort.range) // todo: substitute vars with arg
                    val arrayInterpDecl = mkFreshFuncDecl("array", sort.range, listOf(sort.domain))
                    interpretations[arrayInterpDecl] = KModel.KFuncInterp(
                        sort = sort.range,
                        vars = listOf(arg),
                        entries = entries,
                        default = default
                    )
                    KModel.KFuncInterp(
                        sort = sort as T,
                        vars = emptyList(),
                        entries = emptyList(),
                        default = mkFunctionAsArray<KSort, KSort>(arrayInterpDecl) as KExpr<T>
                    )
                }
                Native.bitwuzlaTermIsFun(term) -> with(converter) {
                    check(decl is KFuncDecl<T>) { "function expected but actual is $decl" }
                    val entries = mutableListOf<KModel.KFuncInterpEntry<T>>()
                    val interp = Native.bitwuzlaGetFunValue(bitwuzlaCtx.bitwuzla, term)
                    check(interp.arity == decl.argSorts.size) {
                        "function arity mismatch: ${interp.arity} and ${decl.argSorts.size}"
                    }
                    for (i in 0 until interp.size) {
                        val args = interp.args[i].zip(decl.argSorts) { arg, sort -> arg.convertExpr(sort) }
                        val value = interp.values[i].convertExpr(decl.sort)
                        entries += KModel.KFuncInterpEntry(args, value)
                    }
                    KModel.KFuncInterp(
                        sort = decl.sort,
                        vars = emptyList(),
                        entries = entries,
                        default = null
                    )
                }
                else -> {
                    val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
                    val convertedValue = with(converter) { value.convertExpr(decl.sort) }
                    KModel.KFuncInterp(
                        sort = decl.sort,
                        vars = emptyList(),
                        entries = emptyList(),
                        default = convertedValue
                    )
                }
            }
        }
    } as KModel.KFuncInterp<T>

    override fun detach(): KModel {
        declarations.forEach { interpretation(it) }
        return KModelImpl(ctx, interpretations.toMutableMap())
    }

    private val valueSampler: KModelEvaluator by lazy {
        val modelStub = KModelImpl(ctx, interpretations = emptyMap())
        KModelEvaluator(ctx, modelStub, true)
    }
}
