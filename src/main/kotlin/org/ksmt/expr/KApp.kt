package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort
import java.util.*


abstract class KApp<T : KSort, A : KExpr<*>> internal constructor(
    val args: List<A>
) : KExpr<T>() {
    abstract fun KContext.decl(): KDecl<T>
    override fun hash(): Int = Objects.hash(javaClass, args)
}

open class KFunctionApp<T : KSort> internal constructor(
    val decl: KDecl<T>, args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(args) {
    override fun KContext.sort(): T = decl.sort
    override fun KContext.decl(): KDecl<T> = decl

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KConst<T : KSort> internal constructor(decl: KDecl<T>) : KFunctionApp<T>(decl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}
