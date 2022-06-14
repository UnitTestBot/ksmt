package org.ksmt.expr

import org.ksmt.decl.*
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


class KAndExpr internal constructor(args: List<KExpr<KBoolSort>>) : KBoolExpr<KExpr<KBoolSort>>(KAndDecl, args) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO()
    }
}

class KOrExpr internal constructor(args: List<KExpr<KBoolSort>>) : KBoolExpr<KExpr<KBoolSort>>(KOrDecl, args) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

}

class KNotExpr internal constructor(val arg: KExpr<KBoolSort>) : KBoolExpr<KExpr<KBoolSort>>(KNotDecl, listOf(arg)) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

}

class KEqExpr<T : KSort> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(KEqDecl, listOf(lhs, rhs)) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

object KTrue : KBoolExpr<KExpr<*>>(KTrueDecl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

object KFalse : KBoolExpr<KExpr<*>>(KFalseDecl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

fun mkAnd(vararg args: KExpr<KBoolSort>) = KAndExpr(args.toList()).intern()
fun mkOr(vararg args: KExpr<KBoolSort>) = KOrExpr(args.toList()).intern()
fun mkNot(arg: KExpr<KBoolSort>) = KNotExpr(arg).intern()
fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>) = KEqExpr(lhs, rhs).intern()

infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
operator fun KExpr<KBoolSort>.not() = mkNot(this)
infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
val Boolean.expr
    get() = if (this) KTrue else KFalse
