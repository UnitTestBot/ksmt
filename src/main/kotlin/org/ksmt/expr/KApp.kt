package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort


abstract class KApp<T : KSort, A : KExpr<*>> internal constructor() : KExpr<T>() {
    abstract val args: List<A>
    abstract fun KContext.decl(): KDecl<T>
}

open class KFunctionApp<T : KSort> internal constructor(
    val decl: KDecl<T>, override val args: List<KExpr<*>>
) : KApp<T, KExpr<*>>() {
    override fun KContext.sort(): T = decl.sort
    override fun KContext.decl(): KDecl<T> = decl

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KConst<T : KSort> internal constructor(decl: KDecl<T>) : KFunctionApp<T>(decl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}
