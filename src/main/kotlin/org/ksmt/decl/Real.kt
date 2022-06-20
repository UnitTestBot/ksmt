package org.ksmt.decl

import org.ksmt.expr.*
import org.ksmt.sort.*

object KRealToIntDecl : KFuncDecl1<KIntSort, KRealSort>("realToInt", KIntSort, KRealSort) {
    override fun apply(arg: KExpr<KRealSort>): KExpr<KIntSort> = mkRealToInt(arg)
}

object KRealIsIntDecl : KFuncDecl1<KBoolSort, KRealSort>("realIsInt", KBoolSort, KRealSort) {
    override fun apply(arg: KExpr<KRealSort>): KExpr<KBoolSort> = mkRealIsInt(arg)
}

class KRealNumDecl(val value: String) : KConstDecl<KRealSort>(value, KRealSort) {
    override fun apply(args: List<KExpr<*>>): KExpr<KRealSort> = mkRealNum(value)
}
