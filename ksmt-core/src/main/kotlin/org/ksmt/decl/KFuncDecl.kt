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
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        return mkFunctionApp(this@KFuncDecl, args)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun toString(): String = buildString {
        append('(')
        append(name)
        append(" (")
        append(argSorts.joinToString(" "))
        append(") ")
        append("$sort")
        append(" )")
    }

    fun checkArgSorts(args: List<KExpr<*>>) = with(ctx) {
        check(args.size == argSorts.size) {
            "${argSorts.size} arguments expected but ${args.size} provided"
        }
        val providedSorts = args.map { it.sort }
        check(providedSorts == argSorts) {
            "Arguments sort mismatch. Expected $argSorts but $providedSorts provided"
        }
    }
}

abstract class KFuncDecl1<T : KSort, A : KSort>(
    ctx: KContext,
    name: String,
    sort: T,
    val argSort: A
) : KFuncDecl<T>(ctx, name, sort, listOf(argSort)) {
    abstract fun KContext.apply(arg: KExpr<A>): KApp<T, KExpr<A>>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
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
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        val (arg0, arg1) = args
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
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        val (arg0, arg1, arg2) = args
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
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        val providedSorts = args.map { it.sort }
        check(providedSorts.all { it == argSort }) {
            "Arguments sort mismatch. Expected arguments of sort $argSort  but $providedSorts provided"
        }
        return applyChain(args as List<KExpr<A>>)
    }
}
