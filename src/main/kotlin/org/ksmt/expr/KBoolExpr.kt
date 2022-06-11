package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort

abstract class KBoolExpr : KExpr<KBoolExpr>() {
    override val sort = KBoolSort
    abstract override val decl: KDecl<KBoolExpr>
}

operator fun KExpr<KBoolExpr>.not() = mkNot(this)
infix fun KExpr<KBoolExpr>.and(other: KExpr<KBoolExpr>) = mkAnd(this, other)
infix fun KExpr<KBoolExpr>.or(other: KExpr<KBoolExpr>) = mkOr(this, other)