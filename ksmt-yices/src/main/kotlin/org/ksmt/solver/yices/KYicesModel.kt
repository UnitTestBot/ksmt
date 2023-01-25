package org.ksmt.solver.yices

import com.sri.yices.Model
import com.sri.yices.YVal
import com.sri.yices.YValTag
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.model.KModelEvaluator
import org.ksmt.solver.model.KModelImpl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.asExpr
import org.ksmt.utils.mkConst
import org.ksmt.utils.mkFreshConstDecl

class KYicesModel(
    private val model: Model,
    private val ctx: KContext,
    private val internalizer: KYicesExprInternalizer,
    private val converter: KYicesExprConverter
) : KModel, AutoCloseable {
    override val declarations: Set<KDecl<*>> by lazy {
        model.collectDefinedTerms().mapTo(hashSetOf()) { converter.convertDecl(it) }
    }

    override val uninterpretedSorts: Set<KUninterpretedSort> by lazy {
        declarations.forEach { interpretation(it) }
        uninterpretedSortsStorage
    }
    private val uninterpretedSortsStorage: MutableSet<KUninterpretedSort> = mutableSetOf()
    private val uninterpretedSortsUniverses = hashMapOf<KUninterpretedSort, MutableSet<KExpr<KUninterpretedSort>>>()

    private val interpretations: MutableMap<KDecl<*>, KModel.KFuncInterp<*>> = hashMapOf()
    private val funcInterpretationsToDo = arrayListOf<Pair<YVal, KFuncDecl<*>>>()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KExpr<KUninterpretedSort>>? {
        if (sort !in uninterpretedSorts) return null

        return uninterpretedSortsUniverses[sort]
    }

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        return KModelEvaluator(ctx, this, isComplete).apply(expr)
    }

    @Suppress("UNCHECKED_CAST")
    private fun getValue(yval: YVal, sort: KSort): KExpr<*> = with(ctx) {
        return when (sort) {
            is KBoolSort -> model.boolValue(yval).expr
            is KBvSort -> mkBv(model.bvValue(yval), sort.sizeBits)
            is KRealSort -> mkRealNum(model.bigRationalValue(yval))
            is KIntSort -> mkRealToInt(mkRealNum(model.bigRationalValue(yval)))
            is KUninterpretedSort -> {
                uninterpretedSortsStorage.add(sort)

                sort.mkConst(model.scalarValue(yval)[0].toString()).also {
                    uninterpretedSortsUniverses.getOrPut(sort) { mutableSetOf() }.add(it)
                }
            }
            is KArraySort<*, *> -> {
                val funcDecl = ctx.mkFreshFuncDecl("array", sort.range, listOf(sort.domain))

                funcInterpretationsToDo.add(Pair(yval, funcDecl))

                mkFunctionAsArray<KSort, KSort>(funcDecl).asExpr(sort)
            }
            else -> error("Unsupported sort $sort")
        }
    }

    private fun <T: KSort> functionInterpretation(yval: YVal, decl: KFuncDecl<T>): KModel.KFuncInterp<T> {
        val functionChildren = model.expandFunction(yval)
        val default = if (yval.tag != YValTag.UNKNOWN)
            getValue(functionChildren.value, decl.sort).asExpr(decl.sort)
        else
            null

        val entries = functionChildren.vector.map { mapping ->
            val entry = model.expandMapping(mapping)
            val args = entry.vector.zip(decl.argSorts).map { (arg, sort) ->
                getValue(arg, sort)
            }
            val res = getValue(entry.value, decl.sort).asExpr(decl.sort)

            KModel.KFuncInterpEntry(args, res)
        }

        return KModel.KFuncInterp(
            decl = decl,
            vars = decl.argSorts.map { it.mkFreshConstDecl("x") },
            entries = entries,
            default = default
        )
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? = with(ctx) {
        interpretations.getOrPut(decl) {
            if (decl !in declarations) return@with null

            val sort = decl.sort
            val yicesDecl = with(internalizer) { decl.internalizeDecl() }
            val yval = model.getValue(yicesDecl)

            val result = when (decl) {
                is KConstDecl<T> -> KModel.KFuncInterp(
                    decl = decl,
                    vars = emptyList(),
                    entries = emptyList(),
                    default = getValue(yval, sort).asExpr(sort)
                )
                is KFuncDecl<T> -> functionInterpretation(yval, decl)
                else -> error("Unexpected declaration $decl")
            }

            while (funcInterpretationsToDo.isNotEmpty()) {
                val (yvalT, declT) = funcInterpretationsToDo.removeLast()

                interpretations[declT] = functionInterpretation(yvalT, declT)
            }

            result
        } as? KModel.KFuncInterp<T>
    }

    override fun detach(): KModel {
        declarations.forEach { interpretation(it) }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        return KModelImpl(ctx, interpretations.toMap(), uninterpretedSortsUniverses)
    }

    override fun close() {
        model.close()
    }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }
}
