package org.ksmt.expr

import org.ksmt.decl.KArithAddDecl
import org.ksmt.decl.KArithGeDecl
import org.ksmt.decl.KArithNumDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort


class KAddArithExpr internal constructor(args: List<KExpr<KArithSort>>) :
    KArithExpr<KExpr<KArithSort>>(KArithAddDecl, args) {
    override fun accept(transformer: KTransformer): KExpr<KArithSort> {
        TODO("Not yet implemented")
    }

}

class KNumArithExpr internal constructor(val value: Int) : KArithExpr<KExpr<*>>(KArithNumDecl, emptyList()) {
    override fun accept(transformer: KTransformer): KExpr<KArithSort> {
        TODO("Not yet implemented")
    }
}

class KGeArithExpr internal constructor(
    val lhs: KExpr<KArithSort>,
    val rhs: KExpr<KArithSort>
) : KBoolExpr<KExpr<KArithSort>>(KArithGeDecl, listOf(lhs, rhs)) {
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

fun mkArithNum(value: Int) = KNumArithExpr(value).intern()
fun mkArithAdd(vararg args: KExpr<KArithSort>) = KAddArithExpr(args.toList()).intern()
fun mkArithGe(lhs: KExpr<KArithSort>, rhs: KExpr<KArithSort>) = KGeArithExpr(lhs, rhs).intern()

operator fun KExpr<KArithSort>.plus(other: KExpr<KArithSort>) = mkArithAdd(this, other)
infix fun KExpr<KArithSort>.ge(other: KExpr<KArithSort>) = mkArithGe(this, other)
val Int.expr: KNumArithExpr
    get() = mkArithNum(this)
