package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KArrayConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

sealed class KArrayStoreBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    val array: KExpr<A>,
    val value: KExpr<R>
) : KApp<A, KSort>(ctx) {
    override val sort: A = array.sort

    abstract val indices: List<KExpr<*>>

    override val args: List<KExpr<KSort>>
        get() = (listOf(array) + indices + listOf(value)).uncheckedCast()
}

class KArrayStore<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    value: KExpr<R>
) : KArrayStoreBase<KArraySort<D, R>, R>(ctx, array, value) {
    override val indices: List<KExpr<*>>
        get() = listOf(index)

    override val decl: KDecl<KArraySort<D, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { index }, { value })
}

class KArray2Store<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray2Sort<D0, D1, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>,
    value: KExpr<R>
) : KArrayStoreBase<KArray2Sort<D0, D1, R>, R>(ctx, array, value) {
    override val indices: List<KExpr<*>>
        get() = listOf(index0, index1)

    override val decl: KDecl<KArray2Sort<D0, D1, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArray2Sort<D0, D1, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int =
        hash(array, index0, index1, value)

    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 }, { value })
}

class KArray3Store<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>,
    val index2: KExpr<D2>,
    value: KExpr<R>
) : KArrayStoreBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, array, value) {
    override val indices: List<KExpr<*>>
        get() = listOf(index0, index1, index2)

    override val decl: KDecl<KArray3Sort<D0, D1, D2, R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArray3Sort<D0, D1, D2, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int =
        hash(array, index0, index1, index2, value)

    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 }, { index2 }, { value })
}

class KArrayNStore<R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArrayNSort<R>>,
    override val indices: List<KExpr<*>>,
    value: KExpr<R>
) : KArrayStoreBase<KArrayNSort<R>, R>(ctx, array, value) {

    override val decl: KDecl<KArrayNSort<R>>
        get() = ctx.mkArrayStoreDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KArrayNSort<R>> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(array, indices, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { indices }, { value })
}

sealed class KArraySelectBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    val array: KExpr<A>
) : KApp<R, KSort>(ctx) {
    override val sort: R = array.sort.range

    abstract val indices: List<KExpr<*>>

    override val args: List<KExpr<KSort>>
        get() = (listOf(array) + indices).uncheckedCast()
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KArraySelectBase<KArraySort<D, R>, R>(ctx, array) {
    override val indices: List<KExpr<*>>
        get() = listOf(index)

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { index })
}

class KArray2Select<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray2Sort<D0, D1, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>
) : KArraySelectBase<KArray2Sort<D0, D1, R>, R>(ctx, array) {
    override val indices: List<KExpr<*>>
        get() = listOf(index0, index1)

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index0, index1)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 })
}

class KArray3Select<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArray3Sort<D0, D1, D2, R>>,
    val index0: KExpr<D0>,
    val index1: KExpr<D1>,
    val index2: KExpr<D2>
) : KArraySelectBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, array) {
    override val indices: List<KExpr<*>>
        get() = listOf(index0, index1, index2)

    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, index0, index1, index2)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { array }, { index0 }, { index1 }, { index2 })
}

class KArrayNSelect<R : KSort> internal constructor(
    ctx: KContext,
    array: KExpr<KArrayNSort<R>>,
    override val indices: List<KExpr<*>>
) : KArraySelectBase<KArrayNSort<R>, R>(ctx, array) {
    override val decl: KDecl<R>
        get() = ctx.mkArraySelectDecl(array.sort)

    override fun accept(transformer: KTransformerBase): KExpr<R> = transformer.transform(this)

    override fun internHashCode(): Int = hash(array, indices)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { array }, { indices })
}

class KArrayConst<A : KArraySortBase<R>, R : KSort> internal constructor(
    ctx: KContext,
    override val sort: A,
    val value: KExpr<R>
) : KApp<A, R>(ctx) {

    override val decl: KArrayConstDecl<A, R>
        get() = ctx.mkArrayConstDecl(sort)

    override val args: List<KExpr<R>>
        get() = listOf(value)

    override fun accept(transformer: KTransformerBase): KExpr<A> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(sort, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { sort }, { value })
}

class KFunctionAsArray<A : KArraySortBase<R>, R : KSort> internal constructor(
    ctx: KContext,
    override val sort: A,
    val function: KFuncDecl<R>
) : KExpr<A>(ctx) {

    init {
        check(function.argSorts == sort.domainSorts) {
            "Function arguments sort mismatch"
        }
    }

    override fun print(printer: ExpressionPrinter): Unit = with(printer) {
        append("(asArray ")
        append(function.name)
        append(")")
    }

    override fun accept(transformer: KTransformerBase): KExpr<A> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(function)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { function }
}

/**
 * Array lambda binding.
 *
 * [Defined as in the Z3 solver](https://microsoft.github.io/z3guide/docs/logic/Lambdas)
 * */
sealed class KArrayLambdaBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    override val sort: A,
    val body: KExpr<R>
) : KExpr<A>(ctx) {
    abstract val indexVarDeclarations: List<KDecl<*>>

    override fun print(printer: ExpressionPrinter) {
        val str = buildString {
            append("(lambda (")

            indexVarDeclarations.forEach {
                append('(')
                append(it.name)
                append(' ')
                it.sort.print(this)
                append(')')
            }

            append(") ")

            body.print(this)

            append(')')
        }
        printer.append(str)
    }
}

class KArrayLambda<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVarDecl: KDecl<D>,
    body: KExpr<R>
) : KArrayLambdaBase<KArraySort<D, R>, R>(
    ctx,
    ctx.mkArraySort(indexVarDecl.sort, body.sort),
    body
) {
    override val indexVarDeclarations: List<KDecl<*>>
        get() = listOf(indexVarDecl)

    override fun accept(transformer: KTransformerBase): KExpr<KArraySort<D, R>> = transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVarDecl, body)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { indexVarDecl }, { body })
}

class KArray2Lambda<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVar0Decl: KDecl<D0>,
    val indexVar1Decl: KDecl<D1>,
    body: KExpr<R>
) : KArrayLambdaBase<KArray2Sort<D0, D1, R>, R>(
    ctx,
    ctx.mkArray2Sort(indexVar0Decl.sort, indexVar1Decl.sort, body.sort),
    body
) {
    override val indexVarDeclarations: List<KDecl<*>>
        get() = listOf(indexVar0Decl, indexVar1Decl)

    override fun accept(transformer: KTransformerBase): KExpr<KArray2Sort<D0, D1, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVar0Decl, indexVar1Decl, body)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { indexVar0Decl }, { indexVar1Decl }, { body })
}

class KArray3Lambda<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    val indexVar0Decl: KDecl<D0>,
    val indexVar1Decl: KDecl<D1>,
    val indexVar2Decl: KDecl<D2>,
    body: KExpr<R>
) : KArrayLambdaBase<KArray3Sort<D0, D1, D2, R>, R>(
    ctx,
    ctx.mkArray3Sort(indexVar0Decl.sort, indexVar1Decl.sort, indexVar2Decl.sort, body.sort),
    body
) {
    override val indexVarDeclarations: List<KDecl<*>>
        get() = listOf(indexVar0Decl, indexVar1Decl, indexVar2Decl)

    override fun accept(transformer: KTransformerBase): KExpr<KArray3Sort<D0, D1, D2, R>> =
        transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVar0Decl, indexVar1Decl, indexVar2Decl, body)
    override fun internEquals(other: Any): Boolean =
        structurallyEqual(other, { indexVar0Decl }, { indexVar1Decl }, { indexVar2Decl }, { body })
}

class KArrayNLambda<R : KSort> internal constructor(
    ctx: KContext,
    override val indexVarDeclarations: List<KDecl<*>>,
    body: KExpr<R>
) : KArrayLambdaBase<KArrayNSort<R>, R>(
    ctx,
    ctx.mkArrayNSort(indexVarDeclarations.map { it.sort }, body.sort),
    body
) {
    override fun accept(transformer: KTransformerBase): KExpr<KArrayNSort<R>> = transformer.transform(this)

    override fun internHashCode(): Int = hash(indexVarDeclarations, body)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { indexVarDeclarations }, { body })
}
