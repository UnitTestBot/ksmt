package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.bitwuzla.bindings.Native
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

    override fun <T : KSort> eval(expr: KExpr<T>, complete: Boolean): KExpr<T> {
        bitwuzlaCtx.ensureActive()
        val term = with(internalizer) { expr.internalize() }
        val value = Native.bitwuzlaGetValue(bitwuzlaCtx.bitwuzla, term)
        return with(converter) { value.convert() }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T> = interpretations.getOrPut(decl) {
        with(ctx) {
            bitwuzlaCtx.ensureActive()
            val term = with(internalizer) {
                bitwuzlaCtx.mkConstant(decl, decl.bitwuzlaSort())
            }
            when {
                Native.bitwuzlaTermIsArray(term) -> with(converter) {
                    val sort = decl.sort as KArraySort<*, *>
                    val entries = mutableListOf<KModel.KFuncInterpEntry<KSort>>()
                    val interp = Native.bitwuzlaGetArrayValue(bitwuzlaCtx.bitwuzla, term)
                    for (i in 0 until interp.size) {
                        val index = interp.indices[i].convert<KSort>()
                        val value = interp.values[i].convert<KSort>()
                        entries += KModel.KFuncInterpEntry(listOf(index), value)
                    }
                    val arg = mkFreshConstDecl("var", sort.domain)
                    val default = interp.defaultValue?.convert<KSort>() // todo: substitute vars with arg
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
                    val entries = mutableListOf<KModel.KFuncInterpEntry<T>>()
                    val interp = Native.bitwuzlaGetFunValue(bitwuzlaCtx.bitwuzla, term)
                    for (i in 0 until interp.size) {
                        val args = interp.args[i].map { it.convert<KSort>() }
                        val value = interp.values[i].convert<T>()
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
                    val convertedValue = with(converter) { value.convert<T>() }
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
}
