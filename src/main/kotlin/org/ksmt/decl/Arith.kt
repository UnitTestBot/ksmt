package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.expr.mkArithAdd
import org.ksmt.expr.mkArithMul
import org.ksmt.expr.mkArithSub
import org.ksmt.expr.mkArithUnaryMinus
import org.ksmt.expr.mkArithDiv
import org.ksmt.expr.mkArithPower
import org.ksmt.expr.mkArithLt
import org.ksmt.expr.mkArithLe
import org.ksmt.expr.mkArithGt
import org.ksmt.expr.mkArithGe
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KArithAddDecl<T : KArithSort<T>>(argumentSort: T) : KFuncDeclChain<T, T>("arithAdd", argumentSort, argumentSort) {
    override fun applyChain(args: List<KExpr<T>>): KExpr<T> = mkArithAdd(args)
}

class KArithMulDecl<T : KArithSort<T>>(argumentSort: T) : KFuncDeclChain<T, T>("arithMul", argumentSort, argumentSort) {
    override fun applyChain(args: List<KExpr<T>>): KExpr<T> = mkArithMul(args)
}

class KArithSubDecl<T : KArithSort<T>>(argumentSort: T) : KFuncDeclChain<T, T>("arithSub", argumentSort, argumentSort) {
    override fun applyChain(args: List<KExpr<T>>): KExpr<T> = mkArithSub(args)
}

class KArithUnaryMinusDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl1<T, T>("arithUnaryMinus", argumentSort, argumentSort) {
    override fun apply(arg: KExpr<T>): KExpr<T> = mkArithUnaryMinus(arg)
}

class KArithDivDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<T, T, T>("arithDiv", argumentSort, argumentSort, argumentSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> = mkArithDiv(arg0, arg1)
}

class KArithPowerDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<T, T, T>("arithPower", argumentSort, argumentSort, argumentSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<T> = mkArithPower(arg0, arg1)
}

class KArithLtDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithLt", KBoolSort, argumentSort, argumentSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> = mkArithLt(arg0, arg1)
}

class KArithLeDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithLe", KBoolSort, argumentSort, argumentSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> = mkArithLe(arg0, arg1)
}

class KArithGtDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithGt", KBoolSort, argumentSort, argumentSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> = mkArithGt(arg0, arg1)
}

class KArithGeDecl<T : KArithSort<T>>(argumentSort: T) :
    KFuncDecl2<KBoolSort, T, T>("arithGe", KBoolSort, argumentSort, argumentSort) {
    override fun apply(arg0: KExpr<T>, arg1: KExpr<T>): KExpr<KBoolSort> = mkArithGe(arg0, arg1)
}
