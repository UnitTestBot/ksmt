package org.ksmt.solver.yices

import com.sri.yices.Model
import com.sri.yices.YVal
import com.sri.yices.YValTag
import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort

class KYicesModel(
    private val model: Model,
    private val ctx: KContext,
    private val internalizer: KYicesExprInternalizer,
    private val converter: KYicesExprConverter
) : KModel {
    override val declarations: Set<KDecl<*>> by lazy {
        model.collectDefinedTerms().mapTo(hashSetOf()) { converter.convertDecl(it) }
    }

    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>?> = hashMapOf()

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        TODO("Not yet implemented")
    }

    private fun boolInterpretation(yval: YVal): KModel.KFuncInterp<KBoolSort> = with(ctx) {
        KModel.KFuncInterp(
            sort = boolSort,
            vars = emptyList(),
            entries = emptyList(),
            default = model.boolValue(yval).expr
        )
    }

    private fun bvInterpretation(yval: YVal, decl: KDecl<KBvSort>): KModel.KFuncInterp<KBvSort> = with(ctx) {
        KModel.KFuncInterp(
            sort = decl.sort,
            vars = emptyList(),
            entries = emptyList(),
            default = mkBv(model.bvValue(yval).toTypedArray(), decl.sort.sizeBits)
        )
    }

    private fun rationalInterpretation(yval: YVal, decl: KDecl<KArithSort<*>>) = with(ctx) {
        val bigRational = model.bigRationalValue(yval)
        val num = mkRealNum(mkIntNum(bigRational.numerator), mkIntNum(bigRational.denominator))

        when (decl.sort) {
            is KIntSort -> KModel.KFuncInterp(
                sort = intSort,
                vars = emptyList(),
                entries = emptyList(),
                default = mkRealToInt(num)
            )
            is KRealSort -> KModel.KFuncInterp(
                sort = realSort,
                vars = emptyList(),
                entries = emptyList(),
                default = num
            )
            else -> error("Unexpected sort ${decl.sort}")
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? =
        interpretations.getOrPut(decl) {
            if (decl !in declarations) return@getOrPut null

            val yicesDecl = with(internalizer) { decl.internalizeDecl() }
            val yval = model.getValue(yicesDecl)

            when (yval.tag) {
                YValTag.BOOL -> boolInterpretation(yval)
                YValTag.RATIONAL -> rationalInterpretation(yval, decl as KDecl<KArithSort<*>>)
                YValTag.BV -> bvInterpretation(yval, decl as KDecl<KBvSort>)
                YValTag.FUNCTION -> TODO()
                else -> TODO("Unsupported tag ${yval.tag}")
            }
        } as? KModel.KFuncInterp<T>

    override fun detach(): KModel {
        TODO("Not yet implemented")
    }
}