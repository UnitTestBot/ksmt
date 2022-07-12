package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort

abstract class KApp<T : KSort, A : KExpr<*>> internal constructor(ctx: KContext) : KExpr<T>(ctx) {
    abstract val args: List<A>
    abstract fun decl(): KDecl<T>
    override fun print(): String = buildString {
        if (args.isEmpty()) {
            with(ctx) { append(decl.name) }
            return@buildString
        }
        append('(')
        with(ctx) {
            append(decl.name)
        }
        for (arg in args) {
            append(' ')
            append("$arg")
        }
        append(')')
    }
}

open class KFunctionApp<T : KSort> internal constructor(
    ctx: KContext,
    val decl: KDecl<T>, override val args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(ctx) {
    override fun sort(): T = decl.sort
    override fun decl(): KDecl<T> = decl

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KConst<T : KSort> internal constructor(ctx: KContext, decl: KDecl<T>) :
    KFunctionApp<T>(ctx, decl, args = emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}
