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

    fun visit(decl: KBvNotDecl): T
    fun visit(decl: KBvReductionAndDecl): T
    fun visit(decl: KBvReductionOrDecl): T
    fun visit(decl: KBvAndDecl): T
    fun visit(decl: KBvOrDecl): T
    fun visit(decl: KBvXorDecl): T
    fun visit(decl: KBvNAndDecl): T
    fun visit(decl: KBvNorDecl): T
    fun visit(decl: KBvXNorDecl): T
    fun visit(decl: KBvNegationDecl): T
    fun visit(decl: KBvAddDecl): T
    fun visit(decl: KBvSubDecl): T
    fun visit(decl: KBvMulDecl): T
    fun visit(decl: KBvUnsignedDivDecl): T
    fun visit(decl: KBvSignedDivDecl): T
    fun visit(decl: KBvUnsignedRemDecl): T
    fun visit(decl: KBvSignedRemDecl): T
    fun visit(decl: KBvSignedModDecl): T
    fun visit(decl: KBvUnsignedLessDecl): T
    fun visit(decl: KBvSignedLessDecl): T
    fun visit(decl: KBvSignedLessOrEqualDecl): T
    fun visit(decl: KBvUnsignedLessOrEqualDecl): T
    fun visit(decl: KBvUnsignedGreaterOrEqualDecl): T
    fun visit(decl: KBvSignedGreaterOrEqualDecl): T
    fun visit(decl: KBvUnsignedGreaterDecl): T
    fun visit(decl: KBvSignedGreaterDecl): T
    fun visit(decl: KConcatDecl): T
    fun visit(decl: KExtractDecl): T
    fun visit(decl: KSignExtDecl): T
    fun visit(decl: KZeroExtDecl): T
    fun visit(decl: KRepeatDecl): T
    fun visit(decl: KBvShiftLeftDecl): T
    fun visit(decl: KBvLogicalShiftRightDecl): T
    fun visit(decl: KBvArithShiftRightDecl): T
    fun visit(decl: KBvRotateLeftDecl): T
    fun visit(decl: KBvRotateLeftIndexedDecl): T
    fun visit(decl: KBvRotateRightDecl): T
    fun visit(decl: KBvRotateRightIndexedDecl): T
    fun visit(decl: KBv2IntDecl): T
    fun visit(decl: KBvAddNoOverflowDecl): T
    fun visit(decl: KBvAddNoUnderflowDecl): T
    fun visit(decl: KBvSubNoOverflowDecl): T
    fun visit(decl: KBvSubNoUnderflowDecl): T
    fun visit(decl: KBvDivNoOverflowDecl): T
    fun visit(decl: KBvNegNoOverflowDecl): T
    fun visit(decl: KBvMulNoOverflowDecl): T
    fun visit(decl: KBvMulNoUnderflowDecl): T
}
