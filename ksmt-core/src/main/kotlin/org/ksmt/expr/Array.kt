package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KArrayConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStore<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    val value: KExpr<R>
) : KApp<KArraySort<D, R>, KExpr<*>>(ctx) {

    override val decl: KDecl<KArraySort<D, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override val args: List<KExpr<*>>
        get() = listOf(array, index, value)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)

    override val sort: KArraySort<D, R>
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): KArraySort<D, R> = array.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += array
    }
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KApp<R, KExpr<*>>(ctx) {

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override val args: List<KExpr<*>>
        get() = listOf(array, index)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override val sort: R
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): R = array.sort.range

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += array
    }
}

class KArrayConst<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    override val sort: KArraySort<D, R>,
    val value: KExpr<R>
) : KApp<KArraySort<D, R>, KExpr<R>>(ctx) {

    override val decl: KArrayConstDecl<D, R>
        get() = ctx.mkArrayConstDecl(sort)

    override val args: List<KExpr<R>>
        get() = listOf(value)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)
}

class KFunctionAsArray<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val function: KFuncDecl<R>
) : KExpr<KArraySort<D, R>>(ctx) {
    val domainSort: D

    init {
        check(function.argSorts.size == 1) {
            "Function with single argument required"
        }
        @Suppress("UNCHECKED_CAST")
        domainSort = function.argSorts.single() as D
    }

    override val sort: KArraySort<D, R>
        get() = ctx.mkArraySort(domainSort, function.sort)

    override fun print(printer: ExpressionPrinter): Unit = with(printer) {
        append("(asArray ")
        append(function.name)
        append(")")
    }

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)
}

/** Array lambda binding.
 *
 * [Defined as in the Z3 solver](https://microsoft.github.io/z3guide/docs/logic/Lambdas)
 * */
class KArrayLambda<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVarDecl: KDecl<D>,
    val body: KExpr<R>
) : KExpr<KArraySort<D, R>>(ctx) {

    override fun print(printer: ExpressionPrinter) {
        val str = buildString {
            append("(lambda ((")
            append(indexVarDecl.name)
            append(' ')

            indexVarDecl.sort.print(this)
            append(")) ")

            body.print(this)

            append(')')
        }
        printer.append(str)
    }

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)

    override val sort: KArraySort<D, R>
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): KArraySort<D, R> = ctx.mkArraySort(indexVarDecl.sort, body.sort)

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += body
    }
}
