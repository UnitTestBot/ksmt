package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KArray2Sort
import org.ksmt.sort.KArray3Sort
import org.ksmt.sort.KArrayNSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KArraySortBase
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

sealed class KArrayStoreDeclBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    sort: A
) : KFuncDecl<A>(ctx, "store", sort, listOf(sort) + sort.domainSorts + listOf(sort.range))

class KArrayStoreDecl<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArraySort<D, R>
) : KArrayStoreDeclBase<KArraySort<D, R>, R>(ctx, sort) {
    override fun apply(args: List<KExpr<*>>): KApp<KArraySort<D, R>, *> {
        val (array, index, value) = args
        return ctx.mkArrayStoreNoSimplify(
            array = array.uncheckedCast(),
            index = index.uncheckedCast(),
            value = value.uncheckedCast()
        )
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArray2StoreDecl<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArray2Sort<D0, D1, R>
) : KArrayStoreDeclBase<KArray2Sort<D0, D1, R>, R>(ctx, sort) {
    override fun apply(args: List<KExpr<*>>): KApp<KArray2Sort<D0, D1, R>, *> {
        val (array, index0, index1, value) = args
        return ctx.mkArrayStoreNoSimplify(
            array = array.uncheckedCast(),
            index0 = index0.uncheckedCast(),
            index1 = index1.uncheckedCast(),
            value = value.uncheckedCast()
        )
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArray3StoreDecl<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArray3Sort<D0, D1, D2, R>
) : KArrayStoreDeclBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, sort) {
    override fun apply(args: List<KExpr<*>>): KApp<KArray3Sort<D0, D1, D2, R>, *> {
        val (array, index0, index1, index2, value) = args
        return ctx.mkArrayStoreNoSimplify(
            array = array.uncheckedCast(),
            index0 = index0.uncheckedCast(),
            index1 = index1.uncheckedCast(),
            index2 = index2.uncheckedCast(),
            value = value.uncheckedCast()
        )
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArrayNStoreDecl<R : KSort> internal constructor(
    ctx: KContext,
    sort: KArrayNSort<R>
) : KArrayStoreDeclBase<KArrayNSort<R>, R>(ctx, sort) {
    override fun apply(args: List<KExpr<*>>): KApp<KArrayNSort<R>, *> =
        ctx.mkArrayNStoreNoSimplify(
            array = args.first().uncheckedCast(),
            indices = args.subList(fromIndex = 1, toIndex = args.size - 1).uncheckedCast(),
            value = args.last().uncheckedCast()
        )

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

sealed class KArraySelectDeclBase<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    sort: A
) : KFuncDecl<R>(ctx, "select", sort.range, listOf(sort) + sort.domainSorts)

class KArraySelectDecl<D : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArraySort<D, R>
) : KArraySelectDeclBase<KArraySort<D, R>, R>(ctx, sort) {

    override fun apply(args: List<KExpr<*>>): KApp<R, *> {
        val (array, index) = args
        return ctx.mkArraySelectNoSimplify(
            array = array.uncheckedCast(),
            index = index.uncheckedCast<_, KExpr<D>>()
        )
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArray2SelectDecl<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArray2Sort<D0, D1, R>
) : KArraySelectDeclBase<KArray2Sort<D0, D1, R>, R>(ctx, sort) {

    override fun apply(args: List<KExpr<*>>): KApp<R, *> {
        val (array, index0, index1) = args
        return ctx.mkArraySelectNoSimplify(
            array = array.uncheckedCast(),
            index0 = index0.uncheckedCast<_, KExpr<D0>>(),
            index1 = index1.uncheckedCast<_, KExpr<D1>>()
        )
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArray3SelectDecl<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext,
    sort: KArray3Sort<D0, D1, D2, R>
) : KArraySelectDeclBase<KArray3Sort<D0, D1, D2, R>, R>(ctx, sort) {

    override fun apply(args: List<KExpr<*>>): KApp<R, *> {
        val (array, index0, index1, index2) = args
        return ctx.mkArraySelectNoSimplify(
            array = array.uncheckedCast(),
            index0 = index0.uncheckedCast<_, KExpr<D0>>(),
            index1 = index1.uncheckedCast<_, KExpr<D1>>(),
            index2 = index2.uncheckedCast<_, KExpr<D2>>()
        )
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArrayNSelectDecl<R : KSort> internal constructor(
    ctx: KContext,
    sort: KArrayNSort<R>
) : KArraySelectDeclBase<KArrayNSort<R>, R>(ctx, sort) {

    override fun apply(args: List<KExpr<*>>): KApp<R, *> =
        ctx.mkArrayNSelectNoSimplify(
            array = args.first().uncheckedCast(),
            indices = args.drop(1)
        )

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KArrayConstDecl<A : KArraySortBase<R>, R : KSort>(
    ctx: KContext,
    sort: A
) : KFuncDecl1<A, R>(ctx, "const", sort, sort.range) {
    override fun KContext.apply(arg: KExpr<R>): KApp<A, R> = mkArrayConst(sort, arg)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}
