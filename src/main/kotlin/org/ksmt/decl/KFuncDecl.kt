package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

open class KFuncDecl<T : KSort>(
    ctx: KContext,
    name: String,
    sort: T,
    val argSorts: List<KSort>
) : KDecl<T>(ctx, name, sort) {
    override fun KContext.apply(args: List<KExpr<*>>): KApp<T, *> {
        check(args.map { it.sort } == argSorts) { "Arguments sort mismatch" }
        return mkFunctionApp(this@KFuncDecl, args)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

abstract class KFuncDecl1<T : KSort, A : KSort>(
    ctx: KContext,
    name: String,
    sort: T,
    val argSort: A
) : KFuncDecl<T>(ctx, name, sort, listOf(argSort)) {
    abstract fun KContext.apply(arg: KExpr<A>): KApp<T, KExpr<A>>

    @Suppress("UNCHECKED_CAST")
    override fun KContext.apply(args: List<KExpr<*>>): KApp<T, *> {
        check(args.size == 1 && args[0].sort == argSort)
        return apply(args[0] as KExpr<A>)
    }
}


abstract class KFuncDecl2<T : KSort, A0 : KSort, A1 : KSort>(
    ctx: KContext,
    name: String,
    sort: T,
    val arg0Sort: A0,
    val arg1Sort: A1
) : KFuncDecl<T>(ctx, name, sort, listOf(arg0Sort, arg1Sort)) {
    abstract fun KContext.apply(arg0: KExpr<A0>, arg1: KExpr<A1>): KApp<T, *>

    @Suppress("UNCHECKED_CAST")
    override fun KContext.apply(args: List<KExpr<*>>): KApp<T, *> {
        check(args.size == 2)
        val (arg0, arg1) = args
        check(arg0.sort == arg0Sort && arg1.sort == arg1Sort)
        return apply(arg0 as KExpr<A0>, arg1 as KExpr<A1>)
    }
}

abstract class KFuncDecl3<T : KSort, A0 : KSort, A1 : KSort, A2 : KSort>(
    ctx: KContext,
    name: String,
    sort: T,
    val arg0Sort: A0,
    val arg1Sort: A1,
    val arg2Sort: A2,
) : KFuncDecl<T>(ctx, name, sort, listOf(arg0Sort, arg1Sort, arg2Sort)) {
    abstract fun KContext.apply(arg0: KExpr<A0>, arg1: KExpr<A1>, arg2: KExpr<A2>): KApp<T, *>

    @Suppress("UNCHECKED_CAST")
    override fun KContext.apply(args: List<KExpr<*>>): KApp<T, *> {
        check(args.size == 3)
        val (arg0, arg1, arg2) = args
        check(arg0.sort == arg0Sort && arg1.sort == arg1Sort && arg2.sort == arg2Sort)
        return apply(arg0 as KExpr<A0>, arg1 as KExpr<A1>, arg2 as KExpr<A2>)
    }
}

abstract class KFuncDeclChain<T : KSort, A : KSort>(
    ctx: KContext,
    name: String,
    sort: T,
    val argSort: A,
) : KFuncDecl<T>(ctx, name, sort, listOf(argSort)) {
    abstract fun KContext.applyChain(args: List<KExpr<A>>): KApp<T, KExpr<A>>

    @Suppress("UNCHECKED_CAST")
    override fun KContext.apply(args: List<KExpr<*>>): KApp<T, *> {
        check(args.all { it.sort == argSort })
        return applyChain(args as List<KExpr<A>>)
    }
}
