package org.ksmt.expr

import org.ksmt.decl.KConstDecl
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

class KFunctionApp<T : KSort> internal constructor(
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

class KConst<T : KSort> internal constructor(decl: KDecl<T>) : KApp<T, KExpr<*>>(decl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>) = KFunctionApp(decl, args).intern()

fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = decl.apply(args)
fun <T : KSort> mkConstApp(decl: KConstDecl<T>) = KConst(decl).intern()
