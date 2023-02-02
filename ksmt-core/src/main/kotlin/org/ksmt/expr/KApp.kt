package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KDecl
import org.ksmt.decl.KParameterizedFuncDecl
import org.ksmt.expr.printer.ExpressionPrinter
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort

abstract class KApp<T : KSort, A : KSort> internal constructor(ctx: KContext) : KExpr<T>(ctx) {

    abstract val args: List<KExpr<A>>

    abstract val decl: KDecl<T>

    @Deprecated("Use property", ReplaceWith("decl"))
    fun decl(): KDecl<T> = decl

    override fun print(printer: ExpressionPrinter) {
        with(printer) {
            if (args.isEmpty()) {
                append(decl)
                return
            }

            append("(")
            append(decl)

            for (arg in args) {
                append(" ")
                append(arg)
            }

            append(")")
        }
    }

    private fun ExpressionPrinter.append(decl: KDecl<*>) {
        append(decl.name)
        if (decl is KParameterizedFuncDecl) {
            append(" (_")
            for (param in decl.parameters) {
                append(" $param")
            }
            append(")")
        }
    }

    override fun internHashCode(): Int = hash(decl, args)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { decl }, { args })
}

open class KFunctionApp<T : KSort> internal constructor(
    ctx: KContext,
    override val decl: KDecl<T>,
    override val args: List<KExpr<KSort>>
) : KApp<T, KSort>(ctx) {
    override val sort: T
        get() = decl.sort

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)
}

class KConst<T : KSort> internal constructor(
    ctx: KContext,
    decl: KDecl<T>
) : KFunctionApp<T>(ctx, decl, args = emptyList()) {
    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)
}
