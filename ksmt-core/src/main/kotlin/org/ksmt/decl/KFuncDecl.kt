package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KFuncDecl<T : KSort>(
    ctx: KContext,
    name: String,
    resultSort: T,
    argSorts: List<KSort>,
) : KDecl<T>(ctx, name, resultSort, argSorts) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

abstract class KFuncDecl1<T : KSort, A : KSort>(
    ctx: KContext,
    name: String,
    resultSort: T,
    val argSort: A,
) : KFuncDecl<T>(ctx, name, resultSort, listOf(argSort)) {
    abstract fun KContext.apply(arg: KExpr<A>): KApp<T, A>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        return apply(args[0] as KExpr<A>)
    }
}


abstract class KFuncDecl2<T : KSort, A0 : KSort, A1 : KSort>(
    ctx: KContext,
    name: String,
    resultSort: T,
    val arg0Sort: A0,
    val arg1Sort: A1,
) : KFuncDecl<T>(ctx, name, resultSort, listOf(arg0Sort, arg1Sort)) {
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
    resultSort: T,
    val arg0Sort: A0,
    val arg1Sort: A1,
    val arg2Sort: A2,
) : KFuncDecl<T>(ctx, name, resultSort, listOf(arg0Sort, arg1Sort, arg2Sort)) {
    abstract fun KContext.apply(arg0: KExpr<A0>, arg1: KExpr<A1>, arg2: KExpr<A2>): KApp<T, *>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        val (arg0, arg1, arg2) = args

        return apply(arg0 as KExpr<A0>, arg1 as KExpr<A1>, arg2 as KExpr<A2>)
    }
}

@Suppress("LongParameterList")
abstract class KFuncDecl4<T : KSort, A0 : KSort, A1 : KSort, A2 : KSort, A3 : KSort>(
    ctx: KContext,
    name: String,
    resultSort: T,
    val arg0Sort: A0,
    val arg1Sort: A1,
    val arg2Sort: A2,
    val arg3Sort: A3
) : KFuncDecl<T>(ctx, name, resultSort, listOf(arg0Sort, arg1Sort, arg2Sort, arg3Sort)) {
    abstract fun KContext.apply(
        arg0: KExpr<A0>,
        arg1: KExpr<A1>,
        arg2: KExpr<A2>,
        arg3: KExpr<A3>
    ): KApp<T, *>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        checkArgSorts(args)
        val (arg0, arg1, arg2, arg3) = args

        return apply(arg0 as KExpr<A0>, arg1 as KExpr<A1>, arg2 as KExpr<A2>, arg3 as KExpr<A3>)
    }
}

abstract class KFuncDeclChain<T : KSort, A : KSort>(
    ctx: KContext,
    name: String,
    resultSort: T,
    val argSort: A,
) : KFuncDecl<T>(ctx, name, resultSort, listOf(argSort)) {
    abstract fun KContext.applyChain(args: List<KExpr<A>>): KApp<T, A>

    override fun print(builder: StringBuilder) {
        builder.append('(')
        builder.append(name)
        builder.append(" (")

        argSort.print(builder)

        builder.append(" *) ")

        sort.print(builder)

        builder.append(" )")
    }

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KApp<T, *> = with(ctx) {
        val providedSorts = args.map { it.sort }
        check(providedSorts.all { it == argSort }) {
            "Arguments sort mismatch. Expected arguments of sort $argSort  but $providedSorts provided"
        }

        return applyChain(args as List<KExpr<A>>)
    }
}
