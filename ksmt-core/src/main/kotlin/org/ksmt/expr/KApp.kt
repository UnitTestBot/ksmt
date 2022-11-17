package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KSort

abstract class KApp<T : KSort, A : KExpr<*>> internal constructor(ctx: KContext) : KExpr<T>(ctx) {
    abstract val args: List<A>

    abstract fun decl(): KDecl<T>

    override fun print(builder: StringBuilder): Unit = with(ctx) {
        with(builder) {
            if (args.isEmpty()) {
                append(decl.name)
                return
            }

            append('(')
            append(decl.name)

            for (arg in args) {
                append(' ')
                arg.print(this)
            }

            append(')')
        }
    }
}

open class KFunctionApp<T : KSort> internal constructor(
    ctx: KContext,
    val decl: KDecl<T>,
    override val args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(ctx) {
    override val sort: T
        get() = decl.sort

    override fun decl(): KDecl<T> = decl

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)
}

class KConst<T : KSort> internal constructor(
    ctx: KContext,
    decl: KDecl<T>
) : KFunctionApp<T>(ctx, decl, args = emptyList()) {
    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)
}
