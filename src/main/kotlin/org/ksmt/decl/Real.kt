package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkRealIsInt
import org.ksmt.expr.mkRealNum
import org.ksmt.expr.mkRealToInt
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort

object KRealToIntDecl : KFuncDecl1<KIntSort, KRealSort>("realToInt", KIntSort, KRealSort) {
    override fun apply(arg: KExpr<KRealSort>): KExpr<KIntSort> = mkRealToInt(arg)
}

object KRealIsIntDecl : KFuncDecl1<KBoolSort, KRealSort>("realIsInt", KBoolSort, KRealSort) {
    override fun apply(arg: KExpr<KRealSort>): KExpr<KBoolSort> = mkRealIsInt(arg)
}

class KRealNumDecl(val value: String) : KConstDecl<KRealSort>(value, KRealSort) {
    override fun apply(args: List<KExpr<*>>): KExpr<KRealSort> = mkRealNum(value)
}
