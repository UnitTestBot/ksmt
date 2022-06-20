package org.ksmt.expr

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KSort
import java.util.*


abstract class KApp<T : KSort, A : KExpr<*>> internal constructor(
    val args: List<A>
) : KExpr<T>() {
    abstract val decl: KDecl<T>
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
    override val decl: KDecl<T>, args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(args) {
    override val sort: T by lazy { decl.sort }

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

internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>) = when {
    args.isEmpty() -> KConst(decl).intern()
    else -> KFunctionApp(decl, args).intern()
}

/*
* For builtin declarations e.g. KAndDecl, mkApp must return the same object as a corresponding builder.
* For example, mkApp(KAndDecl, a, b) and mkAnd(a, b) must end up with the same KAndExpr object.
* To achieve such behaviour we override apply for all builtin declarations.
*/
fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = decl.apply(args)
fun <T : KSort> mkConstApp(decl: KConstDecl<T>) = KConst(decl).intern()
