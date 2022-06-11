package org.ksmt.expr

import org.ksmt.decl.KAddArithDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KGeArithDecl
import org.ksmt.decl.KNumArithDecl


class KAddArithExpr internal constructor(override val args: List<KExpr<KArithExpr>>) : KArithExpr() {
    override val decl = KAddArithDecl
}

class KNumArithExpr internal constructor(val value: Int) : KArithExpr() {
    override val decl = KNumArithDecl
    override val args = emptyList<KExpr<*>>()
}

class KGeArithExpr internal constructor(
    val lhs: KExpr<KArithExpr>,
    val rhs: KExpr<KArithExpr>
) : KBoolExpr() {
    override val decl = KGeArithDecl
    override val args = listOf(lhs, rhs)
}

class KArithConst(override val decl: KConstDecl<KArithExpr>) : KArithExpr() {
    override val args = emptyList<KExpr<*>>()
}

fun mkArithNum(value: Int) = KNumArithExpr(value)

val Int.expr
    get() = mkArithNum(this)

fun mkArithAdd(vararg args: KExpr<KArithExpr>) = KAddArithExpr(args.toList())
fun mkArithGe(lhs: KExpr<KArithExpr>, rhs: KExpr<KArithExpr>) = KGeArithExpr(lhs, rhs)
fun mkArithConst(decl: KConstDecl<KArithExpr>) = KArithConst(decl)
