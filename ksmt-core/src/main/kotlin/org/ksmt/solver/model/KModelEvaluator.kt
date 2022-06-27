package org.ksmt.solver.model

import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.expr.rewrite.KExprSubstitutor
import org.ksmt.solver.KModel
import org.ksmt.sort.KSortVisitor
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KBVSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

open class KModelEvaluator(override val ctx: KContext, val model: KModel, val complete: Boolean) : KTransformer {
    val exprStack = arrayListOf<KExpr<*>>()
    val evaluatedExpressions = hashMapOf<KExpr<*>, KExpr<*>>()
    val evaluatedFunctions = hashMapOf<KExpr<*>, KExpr<*>>()

    override fun <T : KSort> transform(expr: KFunctionApp<T>): KExpr<T> = expr.evalFunction()
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = expr.evalFunction()

    override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> = expr.evalExpr { expr }
    override fun <T : KSort> transformApp(expr: KApp<T, *>): KExpr<T> = expr.evalApp {
        mkApp(expr.decl, it)
    }

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = expr.evalApp { args ->
        val filteredArgs = arrayListOf<KExpr<KBoolSort>>()
        for (arg in args) {
            if (arg == trueExpr) continue
            if (arg == falseExpr) return@evalApp falseExpr
            filteredArgs.add(arg)
        }
        if (filteredArgs.isEmpty()) return@evalApp trueExpr
        mkAnd(filteredArgs)
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = expr.evalApp { args ->
        val filteredArgs = arrayListOf<KExpr<KBoolSort>>()
        for (arg in args) {
            if (arg == falseExpr) continue
            if (arg == trueExpr) return@evalApp trueExpr
            filteredArgs.add(arg)
        }
        if (filteredArgs.isEmpty()) return@evalApp falseExpr
        mkOr(filteredArgs)
    }

    override fun transform(expr: KNotExpr): KExpr<KBoolSort> = expr.evalApp { args ->
        when (val arg = args[0]) {
            trueExpr -> falseExpr
            falseExpr -> trueExpr
            else -> mkNot(arg)
        }
    }

    override fun <T : KSort> transform(expr: KEqExpr<T>): KExpr<KBoolSort> = expr.evalApp { args ->
        val (arg0, arg1) = args
        if (arg0 == arg1) return@evalApp trueExpr
        mkEq(arg0, arg1)
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> = expr.evalApp { args ->
        val (arg0, arg1, arg2) = args
        arg0 as KExpr<KBoolSort>
        arg1 as KExpr<T>
        arg2 as KExpr<T>
        when (arg0) {
            trueExpr -> arg1
            falseExpr -> arg2
            else -> mkIte(arg0, arg1, arg2)
        }
    }

    override fun transformIntNum(expr: KIntNumExpr): KExpr<KIntSort> = expr.evalExpr { expr }
    override fun transform(expr: KRealNumExpr): KExpr<KRealSort> = expr.evalExpr { expr }
    override fun transform(expr: KTrue): KExpr<KBoolSort> = expr.evalExpr { expr }
    override fun transform(expr: KFalse): KExpr<KBoolSort> = expr.evalExpr { expr }

    fun <T : KSort> apply(expr: KExpr<T>): KExpr<T> {
        exprStack.add(expr)
        while (exprStack.isNotEmpty()) {
            val e = exprStack.removeLast()
            e.accept(this)
        }
        return expr.evaluated() ?: error("evaluation failed")
    }

    inline fun <T : KSort> KExpr<T>.evalExpr(eval: () -> KExpr<T>?): KExpr<T> {
        val current = evaluated()
        if (current != null) return current
        val expr = eval() ?: return this
        evaluatedExpressions[this] = expr
        return expr
    }

    @Suppress("UNCHECKED_CAST")
    inline fun <T : KSort, A : KExpr<*>> KApp<T, A>.evalApp(eval: KContext.(args: List<A>) -> KExpr<T>?) =
        evalExpr {
            val evaluatedArgs = arrayListOf<KExpr<*>>()
            val notEvaluatedArgs = arrayListOf<KExpr<*>>()
            for (arg in args) {
                val value = arg.evaluated()
                if (value != null) {
                    evaluatedArgs.add(value)
                    continue
                }
                notEvaluatedArgs.add(arg)
            }
            if (notEvaluatedArgs.isNotEmpty()) {
                exprStack.add(this)
                notEvaluatedArgs.forEach { exprStack.add(it) }
                return@evalExpr null
            }
            ctx.eval(evaluatedArgs as List<A>)
        }

    fun <T : KSort, A : KExpr<*>> KApp<T, A>.evalFunction(): KExpr<T> = evalApp { args ->
        val value = evalFunction(args)
        if (this@evalFunction == value) return@evalApp value
        val evaluatedValue = value.evaluated()
        if (evaluatedValue != null) return@evalApp evaluatedValue
        exprStack.add(this@evalFunction)
        exprStack.add(value)
        null
    }

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort, A : KExpr<*>> KApp<T, A>.evalFunction(args: List<KExpr<*>>): KExpr<T> =
        evaluatedFunctions.getOrPut(this) {
            with(ctx) {
                val interpretation = model.interpretation(decl)
                if (interpretation == null && !complete) {
                    return@getOrPut mkApp(decl, args)
                }
                if (interpretation == null) {
                    return@getOrPut sort.sampleValue()
                }
                check(args.size == interpretation.vars.size)
                evalFuncInterp(interpretation, args)
            }
        } as KExpr<T>

    @Suppress("UNCHECKED_CAST")
    open fun <T : KSort> T.sampleValue(): KExpr<T> = with(ctx) {
        accept(object : KSortVisitor<KExpr<T>> {
            override fun visit(sort: KBoolSort): KExpr<T> = trueExpr as KExpr<T>
            override fun visit(sort: KIntSort): KExpr<T> = 0.intExpr as KExpr<T>
            override fun visit(sort: KRealSort): KExpr<T> = mkRealNum(0) as KExpr<T>
            override fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): KExpr<T> =
                mkArrayConst(sort, sort.range.sampleValue()) as KExpr<T>

            override fun <S : KBVSize> visit(sort: KBVSort<S>): KExpr<T> {
                TODO("Not yet implemented")
            }
        })
    }

    @Suppress("UNCHECKED_CAST")
    open fun <T : KSort> evalFuncInterp(interpretation: KModel.KFuncInterp<T>, args: List<KExpr<*>>): KExpr<T> =
        with(ctx) {
            val varSubstitution = KExprSubstitutor(ctx).apply {
                interpretation.vars.zip(args).forEach { (v, a) ->
                    substitute(mkApp(v, emptyList()) as KExpr<KSort>, a as KExpr<KSort>)
                }
            }
            val entries = interpretation.entries.map { entry ->
                KModel.KFuncInterpEntry(
                    entry.args.map { varSubstitution.apply(it) },
                    varSubstitution.apply(entry.value)
                )
            }
            val default = varSubstitution.apply(interpretation.default)
            return entries.foldRight(default) { entry, acc ->
                val argBinding = mkAnd(entry.args.zip(args) { ea, a -> mkEq(ea as KExpr<KSort>, a as KExpr<KSort>) })
                mkIte(argBinding, entry.value, acc)
            }
        }

    @Suppress("UNCHECKED_CAST")
    fun <T : KSort> KExpr<T>.evaluated(): KExpr<T>? = evaluatedExpressions[this] as? KExpr<T>
}
