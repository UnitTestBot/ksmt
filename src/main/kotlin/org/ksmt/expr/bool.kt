package org.ksmt.expr

import org.ksmt.decl.*
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


class KAndExpr internal constructor(override val args: List<KExpr<KBoolSort>>) : KBoolExpr() {
    override val decl = KAndDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        transformer.visit(this)
        val a = 3
    }
}

class KOrExpr internal constructor(override val args: List<KExpr<KBoolSort>>) : KBoolExpr() {
    override val decl = KOrDecl
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        transformer.visit(this)
    }
}

class KNotExpr internal constructor(val arg: KExpr<KBoolSort>) : KBoolExpr() {
    override val decl = KNotDecl
    override val args = listOf(arg)
}

class KBoolConst internal constructor(override val decl: KConstDecl<KBoolSort>) : KBoolExpr() {
    override val args = emptyList<KExpr<*>>()
}

class KEqExpr<T : KSort> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr() {
    override val decl = KEqDecl
    override val args = listOf(lhs, rhs)
}

object KTrue : KBoolExpr() {
    override val decl = KTrueDecl
    override val args = emptyList<KExpr<*>>()
}

object KFalse : KBoolExpr() {
    override val decl = KFalseDecl
    override val args = emptyList<KExpr<*>>()
}

fun mkAnd(vararg args: KExpr<KBoolSort>) = KAndExpr(args.toList()).intern()
fun mkOr(vararg args: KExpr<KBoolSort>) = KOrExpr(args.toList()).intern()
fun mkNot(arg: KExpr<KBoolSort>) = KNotExpr(arg).intern()
fun mkBoolConst(decl: KConstDecl<KBoolSort>) = KBoolConst(decl).intern()
fun <T : KSort> mkEq(lhs: KExpr<T>, rhs: KExpr<T>) = KEqExpr(lhs, rhs).intern()

infix fun <T : KSort> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
operator fun KExpr<KBoolSort>.not() = mkNot(this)
infix fun KExpr<KBoolSort>.and(other: KExpr<KBoolSort>) = mkAnd(this, other)
infix fun KExpr<KBoolSort>.or(other: KExpr<KBoolSort>) = mkOr(this, other)
val Boolean.expr: KBoolExpr
    get() = if (this) KTrue else KFalse
