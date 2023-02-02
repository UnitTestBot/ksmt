package org.ksmt.decl

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KDecl<T : KSort>(
    ctx: KContext,
    val name: String,
    val sort: T,
    val argSorts: List<KSort>
) : KAst(ctx) {
    abstract fun apply(args: List<KExpr<*>>): KApp<T, *>
    abstract fun <R> accept(visitor: KDeclVisitor<R>): R

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append('(')
        append(name)

        if (this@KDecl is KParameterizedFuncDecl) {
            append(parameters.joinToString(separator = " ", prefix = " [", postfix = "]"))
        }

        argSorts.joinTo(this, separator = " ", prefix = " (", postfix = ") ")

        sort.print(this)
        append(')')
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

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as KDecl<*>

        if (name != other.name) return false
        if (sort != other.sort) return false
        if (argSorts != other.argSorts) return false
        if (this is KParameterizedFuncDecl && parameters != (other as? KParameterizedFuncDecl)?.parameters) return false

        return true
    }

    override fun hashCode(): Int =
        hash(javaClass, name, sort, argSorts, (this as? KParameterizedFuncDecl)?.parameters)
}
