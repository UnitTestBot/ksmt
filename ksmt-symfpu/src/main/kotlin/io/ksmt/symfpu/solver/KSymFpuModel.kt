package io.ksmt.symfpu.solver

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KArrayConst
import io.ksmt.expr.KArrayLambdaBase
import io.ksmt.expr.KArrayStoreBase
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFunctionAsArray
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.expr.transformer.KExprVisitResult
import io.ksmt.expr.transformer.KNonRecursiveVisitor
import io.ksmt.solver.KModel
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.solver.model.KFuncInterpEntryVarsFree
import io.ksmt.solver.model.KFuncInterpEntryWithVars
import io.ksmt.solver.model.KFuncInterpVarsFree
import io.ksmt.solver.model.KFuncInterpWithVars
import io.ksmt.solver.model.KModelEvaluator
import io.ksmt.solver.model.KModelImpl
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort
import io.ksmt.symfpu.operations.pack
import io.ksmt.utils.asExpr
import io.ksmt.utils.uncheckedCast

class KSymFpuModel(underlyingModel: KModel, val ctx: KContext, val transformer: FpToBvTransformer) : KModel {
    private var kModel: KModel = underlyingModel

    override val declarations: Set<KDecl<*>> by lazy {
        kModel.declarations.mapTo(hashSetOf()) { transformer.findFpDeclByMappedDecl(it) ?: it }
    }

    override val uninterpretedSorts
        get() = kModel.uninterpretedSorts

    private val evaluatorWithModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = true) }
    private val evaluatorWithoutModelCompletion by lazy { KModelEvaluator(ctx, this, isComplete = false) }
    private val functionAsArrayVisitor = FunctionAsArrayVisitor()
    private val interpretations: MutableMap<KDecl<*>, KFuncInterp<*>> = hashMapOf()

    override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? =
        kModel.uninterpretedSortUniverse(sort)

    override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> {
        ctx.ensureContextMatch(expr)

        val evaluator = if (isComplete) evaluatorWithModelCompletion else evaluatorWithoutModelCompletion
        return evaluator.apply(expr)
    }

    override fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>? = with(ctx) {
        ensureContextMatch(decl)
        return interpretations.getOrPut(decl) {
            val mappedDecl = transformer.findMappedDeclForFpDecl(decl)
            if (mappedDecl == null) {
                val interpretation = kModel.interpretation<T>(decl)
                interpretation?.let { functionAsArrayVisitor.visitInterpretation(it) }
                return@getOrPut interpretation ?: return null
            }

            val interpretation = kModel.interpretation(mappedDecl) ?: return null
            transformInterpretation(interpretation, decl)
        }.uncheckedCast()
    }

    private fun <T : KSort> transformInterpretation(
        interpretation: KFuncInterp<*>, originalDecl: KDecl<T>,
    ): KFuncInterp<T> = when (interpretation) {
        is KFuncInterpVarsFree<*> -> transformVarsFreeInterpretation(interpretation, originalDecl)
        is KFuncInterpWithVars<*> -> transformInterpretationWithVars(interpretation, originalDecl)
    }

    private fun <T : KSort> transformVarsFreeInterpretation(
        interpretation: KFuncInterpVarsFree<*>, originalDecl: KDecl<T>,
    ): KFuncInterpVarsFree<T> {
        val entries = interpretation.entries.map { entry ->
            val args = originalDecl.argSorts.zip(entry.args) { sort, arg ->
                transformExpr(sort, arg, substitution = emptyMap())
            }
            val value = transformExpr(originalDecl.sort, entry.value, substitution = emptyMap())
            KFuncInterpEntryVarsFree.create(args, value)
        }
        val default = interpretation.default?.let {
            transformExpr(originalDecl.sort, it, substitution = emptyMap())
        }
        return KFuncInterpVarsFree(originalDecl, entries, default)
    }

    private fun <T : KSort> transformInterpretationWithVars(
        interpretation: KFuncInterpWithVars<*>, originalDecl: KDecl<T>,
    ): KFuncInterpWithVars<T> {
        val varsSubstitution = mutableMapOf<KExpr<*>, KExpr<*>>()
        val transformedVars = transformDecls(interpretation.vars, originalDecl.argSorts, varsSubstitution)
        val entries = interpretation.entries.map { entry ->
            val args = originalDecl.argSorts.zip(entry.args) { sort, arg ->
                transformExpr(sort, arg, varsSubstitution)
            }
            val value = transformExpr(originalDecl.sort, entry.value, varsSubstitution)
            KFuncInterpEntryWithVars.create(args, value)
        }
        val default = interpretation.default?.let {
            transformExpr(originalDecl.sort, it, varsSubstitution)
        }
        return KFuncInterpWithVars(originalDecl, transformedVars, entries, default)
    }

    private fun <T : KSort> transformExpr(
        sort: T,
        expr: KExpr<*>,
        substitution: Map<KExpr<*>, KExpr<*>>
    ): KExpr<T> {
        if (expr.sort == sort) return expr.asExpr(sort)

        val substitutedExpr = substitution[expr]
        if (substitutedExpr != null) {
            return substitutedExpr.asExpr(sort)
        }

        if (sort is KFpSort) {
            check(expr.sort is KBvSort) { "Incorrect expr: $expr" }
            return ctx.pack(expr.uncheckedCast(), sort).uncheckedCast()
        }

        if (sort is KArraySortBase<*>) {
            return transformArrayExpr(sort, expr, substitution).uncheckedCast()
        }

        error("Unexpected sort: $sort")
    }

    private fun <R : KSort> transformArrayExpr(
        sort: KArraySortBase<R>,
        expr: KExpr<*>,
        substitution: Map<KExpr<*>, KExpr<*>> = emptyMap()
    ): KExpr<out KArraySortBase<R>> = when (expr) {
        is KFunctionAsArray<*, *> -> transformFunctionAsArrayExpr(sort, expr)
        is KArrayConst<*, *> -> transformArrayConstExpr(sort, expr, substitution)
        is KArrayStoreBase<*, *> -> transformArrayStoreExpr(sort, expr, substitution)
        is KArrayLambdaBase<*, *> -> transformArrayLambdaExpr(sort, expr, substitution)
        else -> error("Unexpected array expr: $expr")
    }

    private fun <R : KSort> transformFunctionAsArrayExpr(
        sort: KArraySortBase<R>,
        expr: KFunctionAsArray<*, *>
    ): KExpr<out KArraySortBase<R>> {
        val functionInterp = kModel.interpretation(expr.function)
            ?: error("No interpretation for as-array: $expr")

        val arrayInterpDecl = ctx.mkFreshFuncDecl("array", sort.range, sort.domainSorts)
        interpretations[arrayInterpDecl] = transformInterpretation(functionInterp, arrayInterpDecl)

        return ctx.mkFunctionAsArray(sort, arrayInterpDecl)
    }

    private fun <R : KSort> transformArrayConstExpr(
        sort: KArraySortBase<R>,
        expr: KArrayConst<*, *>,
        substitution: Map<KExpr<*>, KExpr<*>> = emptyMap()
    ): KExpr<out KArraySortBase<R>> {
        val transformedValue = transformExpr(sort.range, expr.value, substitution)
        return ctx.mkArrayConst(sort, transformedValue)
    }

    private fun <R : KSort> transformArrayStoreExpr(
        sort: KArraySortBase<R>,
        expr: KArrayStoreBase<*, *>,
        substitution: Map<KExpr<*>, KExpr<*>> = emptyMap()
    ): KExpr<out KArraySortBase<R>> {
        val transformedArray = transformArrayExpr(sort, expr.array, substitution)
        val transformedValue = transformExpr(sort.range, expr.value, substitution)
        val transformedIndices = expr.indices.zip(sort.domainSorts) { idx, idxSort ->
            transformExpr(idxSort, idx, substitution)
        }

        return when (sort) {
            is KArraySort<*, *> -> {
                val index = transformedIndices.single()
                ctx.mkArrayStore(transformedArray.uncheckedCast(), index, transformedValue)
            }

            is KArray2Sort<*, *, *> -> {
                val (index0, index1) = transformedIndices
                ctx.mkArrayStore(transformedArray.uncheckedCast(), index0, index1, transformedValue)
            }

            is KArray3Sort<*, *, *, *> -> {
                val (index0, index1, index2) = transformedIndices
                ctx.mkArrayStore(transformedArray.uncheckedCast(), index0, index1, index2, transformedValue)
            }

            is KArrayNSort<*> -> {
                ctx.mkArrayNStore(transformedArray.uncheckedCast(), transformedIndices, transformedValue)
            }
        }
    }

    private fun <R : KSort> transformArrayLambdaExpr(
        sort: KArraySortBase<R>,
        expr: KArrayLambdaBase<*, *>,
        substitution: Map<KExpr<*>, KExpr<*>> = emptyMap()
    ): KExpr<out KArraySortBase<R>> {
        val lambdaBodySubstitution = substitution.toMutableMap()
        val indexDecls = transformDecls(expr.indexVarDeclarations, sort.domainSorts, lambdaBodySubstitution)

        val transformedBody = transformExpr(sort.range, expr.body, lambdaBodySubstitution)

        return when (sort) {
            is KArraySort<*, *> -> {
                val index = indexDecls.single()
                ctx.mkArrayLambda(index, transformedBody)
            }

            is KArray2Sort<*, *, *> -> {
                val (index0, index1) = indexDecls
                ctx.mkArrayLambda(index0, index1, transformedBody)
            }

            is KArray3Sort<*, *, *, *> -> {
                val (index0, index1, index2) = indexDecls
                ctx.mkArrayLambda(index0, index1, index2, transformedBody)
            }

            is KArrayNSort<*> -> {
                ctx.mkArrayNLambda(indexDecls, transformedBody)
            }
        }
    }

    private fun transformDecls(
        decls: List<KDecl<*>>,
        sorts: List<KSort>,
        substitution: MutableMap<KExpr<*>, KExpr<*>>
    ): List<KDecl<*>> {
        val transformedDecls = arrayListOf<KDecl<*>>()
        for ((decl, sort) in decls.zip(sorts)) {
            if (decl.sort == sort) {
                transformedDecls.add(decl)
                continue
            }

            val newDecl = ctx.mkFreshConstDecl(decl.name, sort)
            transformedDecls.add(newDecl)
            substitution[decl.apply(emptyList())] = newDecl.apply()
        }
        return transformedDecls
    }

    override fun detach(): KModel {
        kModel = kModel.detach()

        declarations.forEach {
            interpretation(it) ?: error("missed interpretation for $it")
        }

        val uninterpretedSortsUniverses = uninterpretedSorts.associateWith {
            uninterpretedSortUniverse(it) ?: error("missed sort universe for $it")
        }

        return KModelImpl(ctx, interpretations.toMap(), uninterpretedSortsUniverses)
    }

    override fun close() {
        kModel.close()
    }

    override fun toString(): String = detach().toString()
    override fun hashCode(): Int = detach().hashCode()
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KModel) return false
        return detach() == other
    }

    private inner class FunctionAsArrayVisitor : KNonRecursiveVisitor<Unit>(ctx) {
        override fun <T : KSort> defaultValue(expr: KExpr<T>) = Unit
        override fun mergeResults(left: Unit, right: Unit) = Unit

        override fun <A : KArraySortBase<R>, R : KSort> visit(expr: KFunctionAsArray<A, R>): KExprVisitResult<Unit> {
            interpretation(expr.function)
            return saveVisitResult(expr, Unit)
        }

        fun <T : KSort> visitInterpretation(interpretation: KFuncInterp<T>) {
            // Non-array expression cannot contain function-as-array
            if (interpretation.sort !is KArraySortBase<*>) return

            interpretation.default?.let { applyVisitor(it) }
            interpretation.entries.forEach { applyVisitor(it.value) }
        }
    }
}
