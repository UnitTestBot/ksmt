package org.ksmt.expr

import org.ksmt.decl.*
import org.ksmt.expr.manager.ExprManager.intern


class KAndExpr internal constructor(override val args: List<KExpr<KBoolExpr>>) : KBoolExpr() {
    override val decl = KAndDecl
}

class KOrExpr internal constructor(override val args: List<KExpr<KBoolExpr>>) : KBoolExpr() {
    override val decl = KOrDecl
}

class KNotExpr internal constructor(val arg: KExpr<KBoolExpr>) : KBoolExpr() {
    override val decl = KNotDecl
    override val args = listOf(arg)
}

class KBoolConst internal constructor(override val decl: KConstDecl<KBoolExpr>): KBoolExpr(){
    override val args = emptyList<KExpr<*>>()
}

class KEqExpr<T : KExpr<T>> internal constructor(
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

fun mkAnd(vararg args: KExpr<KBoolExpr>) = KAndExpr(args.toList()).intern()
fun mkOr(vararg args: KExpr<KBoolExpr>) = KOrExpr(args.toList()).intern()
fun mkNot(arg: KExpr<KBoolExpr>) = KNotExpr(arg).intern()
fun mkBoolConst(decl: KConstDecl<KBoolExpr>) = KBoolConst(decl).intern()
fun <T : KExpr<T>> mkEq(lhs: KExpr<T>, rhs: KExpr<T>) = KEqExpr(lhs, rhs).intern()