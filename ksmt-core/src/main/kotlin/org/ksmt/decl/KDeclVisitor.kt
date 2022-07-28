package org.ksmt.decl

import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort

interface KDeclVisitor<T> {
    fun <S : KSort> visit(decl: KDecl<S>): Any = error("visitor is not implemented for decl $decl")

    fun <S : KSort> visit(decl: KFuncDecl<S>): T
    fun <S : KSort> visit(decl: KConstDecl<S>): T = visit(decl as KFuncDecl<S>)

    fun visit(decl: KAndDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KNotDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KOrDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KImpliesDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KXorDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KFalseDecl): T = visit(decl as KConstDecl<KBoolSort>)
    fun visit(decl: KTrueDecl): T = visit(decl as KConstDecl<KBoolSort>)
    fun <S : KSort> visit(decl: KEqDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KSort> visit(decl: KDistinctDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KSort> visit(decl: KIteDecl<S>): T = visit(decl as KFuncDecl<S>)

    fun <D : KSort, R : KSort> visit(decl: KArraySelectDecl<D, R>): T = visit(decl as KFuncDecl<R>)
    fun <D : KSort, R : KSort> visit(decl: KArrayStoreDecl<D, R>): T = visit(decl as KFuncDecl<KArraySort<D, R>>)
    fun <D : KSort, R : KSort> visit(decl: KArrayConstDecl<D, R>): T = visit(decl as KFuncDecl<KArraySort<D, R>>)

    fun <S : KArithSort<S>> visit(decl: KArithSubDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort<S>> visit(decl: KArithMulDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort<S>> visit(decl: KArithUnaryMinusDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort<S>> visit(decl: KArithPowerDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort<S>> visit(decl: KArithAddDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort<S>> visit(decl: KArithDivDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort<S>> visit(decl: KArithGeDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KArithSort<S>> visit(decl: KArithGtDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KArithSort<S>> visit(decl: KArithLeDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KArithSort<S>> visit(decl: KArithLtDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)

    fun visit(decl: KIntModDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun visit(decl: KIntRemDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun visit(decl: KIntToRealDecl): T = visit(decl as KFuncDecl<KRealSort>)
    fun visit(decl: KIntNumDecl): T = visit(decl as KConstDecl<KIntSort>)

    fun visit(decl: KRealIsIntDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KRealToIntDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun visit(decl: KRealNumDecl): T = visit(decl as KConstDecl<KRealSort>)

    fun visit(decl: KBitVec1ValueDecl): T
    fun visit(decl: KBitVec8ValueDecl): T
    fun visit(decl: KBitVec16ValueDecl): T
    fun visit(decl: KBitVec32ValueDecl): T
    fun visit(decl: KBitVec64ValueDecl): T
    fun visit(decl: KBitVecCustomSizeValueDecl): T

    fun <S: KBvSort> visit(decl: KBvNotDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvReductionAndDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvReductionOrDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvAndDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvOrDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvXorDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvNAndDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvNorDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvXNorDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvNegationDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvAddDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSubDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvMulDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvUnsignedDivDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedDivDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvUnsignedRemDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedRemDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedModDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvUnsignedLessDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedLessDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedLessOrEqualDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvUnsignedLessOrEqualDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvUnsignedGreaterOrEqualDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedGreaterOrEqualDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvUnsignedGreaterDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSignedGreaterDecl<S>): T
    fun visit(decl: KBvConcatDecl): T
    fun visit(decl: KBvExtractDecl): T
    fun visit(decl: KSignExtDecl): T
    fun visit(decl: KZeroExtDecl): T
    fun visit(decl: KBvRepeatDecl): T
    fun <S : KBvSort> visit(decl: KBvShiftLeftDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvLogicalShiftRightDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvArithShiftRightDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvRotateLeftDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvRotateLeftIndexedDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvRotateRightDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvRotateRightIndexedDecl<S>): T
    fun visit(decl: KBv2IntDecl): T
    fun <S : KBvSort> visit(decl: KBvAddNoOverflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvAddNoUnderflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSubNoOverflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvSubNoUnderflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvDivNoOverflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvNegNoOverflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvMulNoOverflowDecl<S>): T
    fun <S : KBvSort> visit(decl: KBvMulNoUnderflowDecl<S>): T
}
