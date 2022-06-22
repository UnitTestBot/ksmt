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
    override fun equalTo(other: KExpr<*>): Boolean {
        if (this === other) return true
        if (javaClass != other.javaClass) return false
        other as KApp<*, *>
        if (args != other.args) return false
        return true
    }
}

open class KFunctionApp<T : KSort> internal constructor(
    val decl: KDecl<T>, args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(args) {
    override fun KContext.sort(): T = decl.sort
    override fun KContext.decl(): KDecl<T> = decl
    override fun equalTo(other: KExpr<*>): Boolean {
        if (!super.equalTo(other)) return false
        other as KFunctionApp<*>
        return decl == other.decl
    }

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KConst<T : KSort> internal constructor(decl: KDecl<T>) : KFunctionApp<T>(decl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}
