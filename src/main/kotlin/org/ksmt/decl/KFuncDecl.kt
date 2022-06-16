package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkFunctionApp
import org.ksmt.sort.KSort
import java.util.*

open class KFuncDecl<T : KSort>(name: String, sort: T, val argSorts: List<KSort>) : KDecl<T>(name, sort) {
    override fun apply(args: List<KExpr<*>>): KExpr<T> {
        check(args.map { it.sort } == argSorts) { "Arguments sort mismatch" }
        return mkFunctionApp(this, args)
    }

    override fun equals(other: Any?): Boolean {
        if (!super.equals(other)) return false
        other as KFuncDecl<*>
        if (argSorts != other.argSorts) return false
        return true
    }

    override fun hashCode(): Int = Objects.hash(super.hashCode(), argSorts)
}

abstract class KFuncDecl1<T : KSort, A : KSort>(name: String, sort: T, val argSort: A) :
    KFuncDecl<T>(name, sort, listOf(argSort)) {
    abstract fun apply(arg: KExpr<A>): KExpr<T>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KExpr<T> {
        check(args.size == 1 && args[0].sort == argSort)
        return apply(args[0] as KExpr<A>)
    }
}


abstract class KFuncDecl2<T : KSort, A0 : KSort, A1 : KSort>(
    name: String,
    sort: T,
    val arg0Sort: A0,
    val arg1Sort: A1
) : KFuncDecl<T>(name, sort, listOf(arg0Sort, arg1Sort)) {
    abstract fun apply(arg0: KExpr<A0>, arg1: KExpr<A1>): KExpr<T>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KExpr<T> {
        check(args.size == 2)
        val (arg0, arg1) = args
        check(arg0.sort == arg0Sort && arg1.sort == arg1Sort)
        return apply(arg0 as KExpr<A0>, arg1 as KExpr<A1>)
    }
}

abstract class KFuncDecl3<T : KSort, A0 : KSort, A1 : KSort, A2 : KSort>(
    name: String,
    sort: T,
    val arg0Sort: A0,
    val arg1Sort: A1,
    val arg2Sort: A2,
) : KFuncDecl<T>(name, sort, listOf(arg0Sort, arg1Sort, arg2Sort)) {
    abstract fun apply(arg0: KExpr<A0>, arg1: KExpr<A1>, arg2: KExpr<A2>): KExpr<T>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KExpr<T> {
        check(args.size == 3)
        val (arg0, arg1, arg2) = args
        check(arg0.sort == arg0Sort && arg1.sort == arg1Sort && arg2.sort == arg2Sort)
        return apply(arg0 as KExpr<A0>, arg1 as KExpr<A1>, arg2 as KExpr<A2>)
    }
}

abstract class KFuncDeclChain<T : KSort, A : KSort>(
    name: String,
    sort: T,
    val argSort: A,
) : KFuncDecl<T>(name, sort, listOf(argSort)) {
    abstract fun applyChain(args: List<KExpr<A>>): KExpr<T>

    @Suppress("UNCHECKED_CAST")
    override fun apply(args: List<KExpr<*>>): KExpr<T> {
        check(args.all { it.sort == argSort })
        return applyChain(args as List<KExpr<A>>)
    }
}

fun <T : KSort> mkFuncDecl(name: String, sort: T, args: List<KSort>) = KFuncDecl(name, sort, args)
