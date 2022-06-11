package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArithSort

abstract class KArithExpr : KExpr<KArithExpr>() {
    override val sort = KArithSort
    abstract override val decl: KDecl<KArithExpr>
}

operator fun KExpr<KArithExpr>.plus(other: KExpr<KArithExpr>) = mkArithAdd(this, other)
infix fun KExpr<KArithExpr>.ge(other: KExpr<KArithExpr>) = mkArithGe(this, other)
