package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort
import java.util.*
import org.ksmt.expr.manager.ExprManager.intern

abstract class KExpr<T : KSort> {
    abstract val sort: T
    abstract val decl: KDecl<T>
    abstract val args: List<KExpr<*>>

    private val hash by lazy(LazyThreadSafetyMode.NONE) { Objects.hash(javaClass, sort, decl, args) }

    override fun equals(other: Any?): Boolean = this === other
    override fun hashCode(): Int = hash

    internal fun equalTo(other: KExpr<*>): Boolean {
        if (javaClass != other.javaClass) return false
        if (decl != other.decl) return false
        if (sort != other.sort) return false
        if (args != other.args) return false
        return true
    }

    abstract fun accept(transformer: KTransformer): KExpr<T>

}

class KApp<T : KSort> internal constructor(override val decl: KDecl<T>, override val args: List<KExpr<*>>): KExpr<T>(){
    override val sort = decl.sort
}

fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = KApp(decl, args).intern()