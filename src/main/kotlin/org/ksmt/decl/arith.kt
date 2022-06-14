package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkArithAdd
import org.ksmt.expr.mkArithGe
import org.ksmt.expr.mkArithNum
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KArithNumDecl(val value: Int) : KBuiltinConstDecl<KArithSort>("int_$value", KArithSort) {
    override fun applyBuiltin(): KExpr<KArithSort> = mkArithNum(value)
}

object KArithAddDecl : KFuncDeclChain<KArithSort, KArithSort>("arithAdd", KArithSort, KArithSort) {
    override fun applyChain(args: List<KExpr<KArithSort>>): KExpr<KArithSort> = mkArithAdd(args)
}

object KArithGeDecl : KFuncDecl2<KBoolSort, KArithSort, KArithSort>("arithGe", KBoolSort, KArithSort, KArithSort) {
    override fun apply(arg0: KExpr<KArithSort>, arg1: KExpr<KArithSort>): KExpr<KBoolSort> = mkArithGe(arg0, arg1)
}
