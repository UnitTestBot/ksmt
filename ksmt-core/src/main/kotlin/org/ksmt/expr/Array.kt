package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KArrayConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStore<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    val value: KExpr<R>
) : KApp<KArraySort<D, R>, KExpr<*>>(ctx) {
    override fun sort(): KArraySort<D, R> = with(ctx) { array.sort }

    override fun decl(): KDecl<KArraySort<D, R>> = with(ctx) { mkArrayStoreDecl(array.sort) }

    override val args: List<KExpr<*>>
        get() = listOf(array, index, value)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KApp<R, KExpr<*>>(ctx) {
    override fun sort(): R = with(ctx) { array.sort.range }

    override fun decl(): KDecl<R> = with(ctx) { mkArraySelectDecl(array.sort) }

    override val args: List<KExpr<*>>
        get() = listOf(array, index)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)
}

class KArrayConst<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val sort: KArraySort<D, R>,
    val value: KExpr<R>
) : KApp<KArraySort<D, R>, KExpr<R>>(ctx) {
    override fun sort(): KArraySort<D, R> = sort

    override fun decl(): KArrayConstDecl<D, R> = ctx.mkArrayConstDecl(sort)

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

    override fun sort(): KArraySort<D, R> = with(ctx) { mkArraySort(domainSort, function.sort) }

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append("(asArray ")
        append(function.name)
        append(')')
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
    override fun sort(): KArraySort<D, R> = with(ctx) { mkArraySort(indexVarDecl.sort, body.sort) }

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append("(lambda ((")
        append(indexVarDecl.name)
        append(' ')

        indexVarDecl.sort.print(this)
        append(")) ")

        body.print(this)

        append(')')
    }

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)
}
