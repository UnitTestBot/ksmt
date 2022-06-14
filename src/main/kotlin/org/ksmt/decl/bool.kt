package org.ksmt.decl

import org.ksmt.expr.*
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

object KAndDecl : KFuncDeclChain<KBoolSort, KBoolSort>("and", KBoolSort, KBoolSort) {
    override fun applyChain(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = mkAnd(args)
}

object KOrDecl : KFuncDeclChain<KBoolSort, KBoolSort>("or", KBoolSort, KBoolSort) {
    override fun applyChain(args: List<KExpr<KBoolSort>>): KExpr<KBoolSort> = mkOr(args)
}

object KNotDecl : KFuncDecl1<KBoolSort, KBoolSort>("not", KBoolSort, KBoolSort) {
    override fun apply(arg: KExpr<KBoolSort>): KExpr<KBoolSort> = mkNot(arg)
}

class KEqDecl<T : KSort>(argSort: T) : KFuncDecl2<KBoolSort, T, T>("eq", KBoolSort, argSort, argSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> = mkEq(arg0, arg1)
}

object KTrueDecl : KBuiltinConstDecl<KBoolSort>("true", KBoolSort) {
    override fun applyBuiltin(): KExpr<KBoolSort> = mkTrue()
}

object KFalseDecl : KBuiltinConstDecl<KBoolSort>("false", KBoolSort) {
    override fun applyBuiltin(): KExpr<KBoolSort> = mkFalse()
}
