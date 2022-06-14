package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KSort
import java.util.*


abstract class KApp<T : KSort, A : KExpr<*>> internal constructor(
    val decl: KDecl<T>,
    val args: List<A>
) : KExpr<T>() {
    override val sort = decl.sort

    override fun hash(): Int = Objects.hash(sort, decl, args)

    override fun equalTo(other: KExpr<*>): Boolean {
        if (other !is KApp<*, *>) return false
        if (decl != other.decl) return false
        if (args != other.args) return false
        return true
    }
}

/* todo: fix problem with app duplication.
*  For example, mkApp(KAndDecl, a, b) and mkAnd(a, b) should return the same object.
* */
class KAppImpl<T : KSort> internal constructor(
    decl: KDecl<T>, args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(decl, args) {
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }

    override fun equalTo(other: KExpr<*>): Boolean {
        if (!super.equalTo(other)) return false
        return javaClass == other.javaClass
    }
}


fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = KAppImpl(decl, args).intern()
