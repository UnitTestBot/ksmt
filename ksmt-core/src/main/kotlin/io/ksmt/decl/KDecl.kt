package io.ksmt.decl

import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort

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

        if (this@KDecl is KStringLiteralDecl) {
            append("\"$name\"")
        } else {
            append(name)
        }

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
