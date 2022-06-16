package org.ksmt.expr

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.expr.transformer.KFunctionTransformer
import org.ksmt.expr.transformer.KTransformer
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

class KFunctionApp<T : KSort> internal constructor(
    override val decl: KDecl<T>, args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(args) {
    override val sort: T by lazy { decl.sort }

    override fun equalTo(other: KExpr<*>): Boolean {
        if (!super.equalTo(other)) return false
        other as KFunctionApp<*>
        return decl == other.decl
    }

    override fun accept(transformer: KTransformer): KExpr<T> {
        transformer as KFunctionTransformer
        val transformedArgs = args.map { it.accept(transformer) }
        if (transformedArgs == args) return transformer.transformFunctionApp(this)
        return transformer.transformFunctionApp(mkFunctionApp(decl, transformedArgs))
    }
}

class KConst<T : KSort> internal constructor(override val decl: KDecl<T>) : KApp<T, KExpr<*>>(emptyList()) {
    override val sort: T by lazy { decl.sort }

    override fun equalTo(other: KExpr<*>): Boolean {
        if (!super.equalTo(other)) return false
        other as KConst<*>
        return decl == other.decl
    }

    override fun accept(transformer: KTransformer): KExpr<T> {
        transformer as KFunctionTransformer
        return transformer.transformConst(this)
    }
}

internal fun <T : KSort> mkFunctionApp(decl: KDecl<T>, args: List<KExpr<*>>) = KFunctionApp(decl, args).intern()

fun <T : KSort> mkApp(decl: KDecl<T>, args: List<KExpr<*>>) = decl.apply(args)
fun <T : KSort> mkConstApp(decl: KConstDecl<T>) = KConst(decl).intern()
