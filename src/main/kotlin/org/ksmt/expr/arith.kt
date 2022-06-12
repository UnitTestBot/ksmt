package org.ksmt.expr

import org.ksmt.decl.KArithAddDecl
import org.ksmt.decl.KArithGeDecl
import org.ksmt.decl.KArithNumDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KArithSort


class KAddArithExpr internal constructor(override val args: List<KExpr<KArithSort>>) : KArithExpr() {
    override val decl = KArithAddDecl
}

class KNumArithExpr internal constructor(val value: Int) : KArithExpr() {
    override val decl = KArithNumDecl
    override val args = emptyList<KExpr<*>>()
}

class KGeArithExpr internal constructor(
    val lhs: KExpr<KArithSort>,
    val rhs: KExpr<KArithSort>
) : KBoolExpr() {
    override val decl = KArithGeDecl
    override val args = listOf(lhs, rhs)
}

class KArithConst(override val decl: KConstDecl<KArithSort>) : KArithExpr() {
    override val args = emptyList<KExpr<*>>()
}

fun mkArithNum(value: Int) = KNumArithExpr(value).intern()
fun mkArithAdd(vararg args: KExpr<KArithSort>) = KAddArithExpr(args.toList()).intern()
fun mkArithGe(lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort>) = KGeArithExpr(lhs, rhs).intern()
fun mkArithConst(decl: KConstDecl<KArithSort>) = KArithConst(decl).intern()

operator fun KExpr<KArithSort>.plus(other: KExpr<KArithSort>) = mkArithAdd(this, other)
infix fun KExpr<KArithSort>.ge(other: KExpr<KArithSort>) = mkArithGe(this, other)
val Int.expr: KNumArithExpr
    get() = mkArithNum(this)
