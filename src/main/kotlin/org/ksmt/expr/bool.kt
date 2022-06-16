package org.ksmt.expr

import org.ksmt.decl.*
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.expr.transformer.KBoolTransformer
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


class KAndExpr internal constructor(args: List<KExpr<KBoolSort>>) : KBoolExpr<KExpr<KBoolSort>>(args) {
    override val decl = KAndDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        transformer as KBoolTransformer
        val transformedArgs = args.map { it.accept(transformer) }
        if (transformedArgs == args) return transformer.transformAnd(this)
        return transformer.transformAnd(mkAnd(transformedArgs))
    }
}

class KOrExpr internal constructor(args: List<KExpr<KBoolSort>>) : KBoolExpr<KExpr<KBoolSort>>(args) {
    override val decl = KOrDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        transformer as KBoolTransformer
        val transformedArgs = args.map { it.accept(transformer) }
        if (transformedArgs == args) return transformer.transformOr(this)
        return transformer.transformOr(mkOr(transformedArgs))
    }

}

class KNotExpr internal constructor(val arg: KExpr<KBoolSort>) : KBoolExpr<KExpr<KBoolSort>>(listOf(arg)) {
    override val decl = KNotDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

}

class KEqExpr<T : KSort> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(listOf(lhs, rhs)) {
    override val decl by lazy { KEqDecl(lhs.sort) }
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

object KTrue : KBoolExpr<KExpr<*>>(emptyList()) {
    override val decl = KTrueDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

object KFalse : KBoolExpr<KExpr<*>>(emptyList()) {
    override val decl = KFalseDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

fun mkAnd(vararg args: KExpr<KBoolSort>) = KAndExpr(args.toList()).intern()
fun mkAnd(args: List<KExpr<KBoolSort>>) = KAndExpr(args).intern()
fun mkOr(vararg args: KExpr<KBoolSort>) = KOrExpr(args.toList()).intern()
fun mkOr(args: List<KExpr<KBoolSort>>) = KOrExpr(args).intern()
fun mkNot(arg: KExpr<KBoolSort>) = KNotExpr(arg).intern()
fun mkTrue() = KTrue
fun mkFalse() = KFalse
fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>) = KEqExpr(lhs, rhs).intern()

infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
operator fun KExpr<KBoolSort>.not() = mkNot(this)
infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
val Boolean.expr
    get() = if (this) mkTrue() else mkFalse()
