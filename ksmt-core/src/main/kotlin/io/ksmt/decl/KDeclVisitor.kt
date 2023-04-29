package io.ksmt.decl

import io.ksmt.sort.KArithSort
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArray3Sort
import io.ksmt.sort.KArrayNSort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KArraySortBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort

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
    fun <D0 : KSort, D1 : KSort, R : KSort> visit(decl: KArray2SelectDecl<D0, D1, R>): T = visit(decl as KFuncDecl<R>)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        decl: KArray3SelectDecl<D0, D1, D2, R>
    ): T = visit(decl as KFuncDecl<R>)

    fun <R : KSort> visit(decl: KArrayNSelectDecl<R>): T = visit(decl as KFuncDecl<R>)

    fun <D : KSort, R : KSort> visit(decl: KArrayStoreDecl<D, R>): T = visit(decl as KFuncDecl<KArraySort<D, R>>)

    fun <D0 : KSort, D1 : KSort, R : KSort> visit(
        decl: KArray2StoreDecl<D0, D1, R>
    ): T = visit(decl as KFuncDecl<KArray2Sort<D0, D1, R>>)

    fun <D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> visit(
        decl: KArray3StoreDecl<D0, D1, D2, R>
    ): T = visit(decl as KFuncDecl<KArray3Sort<D0, D1, D2, R>>)

    fun <R : KSort> visit(decl: KArrayNStoreDecl<R>): T = visit(decl as KFuncDecl<KArrayNSort<R>>)

    fun <A : KArraySortBase<R>, R : KSort> visit(decl: KArrayConstDecl<A, R>): T = visit(decl as KFuncDecl<A>)

    fun <S : KArithSort> visit(decl: KArithSubDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort> visit(decl: KArithMulDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort> visit(decl: KArithUnaryMinusDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort> visit(decl: KArithPowerDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort> visit(decl: KArithAddDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort> visit(decl: KArithDivDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KArithSort> visit(decl: KArithGeDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KArithSort> visit(decl: KArithGtDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KArithSort> visit(decl: KArithLeDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KArithSort> visit(decl: KArithLtDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)

    fun visit(decl: KIntModDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun visit(decl: KIntRemDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun visit(decl: KIntToRealDecl): T = visit(decl as KFuncDecl<KRealSort>)
    fun visit(decl: KIntNumDecl): T = visit(decl as KConstDecl<KIntSort>)

    fun visit(decl: KRealIsIntDecl): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KRealToIntDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun visit(decl: KRealNumDecl): T = visit(decl as KConstDecl<KRealSort>)

    fun visit(decl: KBitVec1ValueDecl): T = visit(decl as KFuncDecl<KBv1Sort>)
    fun visit(decl: KBitVec8ValueDecl): T = visit(decl as KFuncDecl<KBv8Sort>)
    fun visit(decl: KBitVec16ValueDecl): T = visit(decl as KFuncDecl<KBv16Sort>)
    fun visit(decl: KBitVec32ValueDecl): T = visit(decl as KFuncDecl<KBv32Sort>)
    fun visit(decl: KBitVec64ValueDecl): T = visit(decl as KFuncDecl<KBv64Sort>)
    fun visit(decl: KBitVecCustomSizeValueDecl): T = visit(decl as KFuncDecl<KBvSort>)

    fun <S : KBvSort> visit(decl: KBvNotDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvReductionAndDecl<S>): T = visit(decl as KFuncDecl<KBv1Sort>)
    fun <S : KBvSort> visit(decl: KBvReductionOrDecl<S>): T = visit(decl as KFuncDecl<KBv1Sort>)
    fun <S : KBvSort> visit(decl: KBvAndDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvOrDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvXorDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvNAndDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvNorDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvXNorDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvNegationDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvAddDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvSubDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvMulDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvUnsignedDivDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvSignedDivDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvUnsignedRemDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvSignedRemDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvSignedModDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvUnsignedLessDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvSignedLessDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvSignedLessOrEqualDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvUnsignedLessOrEqualDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvUnsignedGreaterOrEqualDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvSignedGreaterOrEqualDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvUnsignedGreaterDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvSignedGreaterDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun visit(decl: KBvConcatDecl): T = visit(decl as KFuncDecl<KBvSort>)
    fun visit(decl: KBvExtractDecl): T = visit(decl as KFuncDecl<KBvSort>)
    fun visit(decl: KSignExtDecl): T = visit(decl as KFuncDecl<KBvSort>)
    fun visit(decl: KZeroExtDecl): T = visit(decl as KFuncDecl<KBvSort>)
    fun visit(decl: KBvRepeatDecl): T = visit(decl as KFuncDecl<KBvSort>)
    fun <S : KBvSort> visit(decl: KBvShiftLeftDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvLogicalShiftRightDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvArithShiftRightDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvRotateLeftDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvRotateLeftIndexedDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvRotateRightDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun <S : KBvSort> visit(decl: KBvRotateRightIndexedDecl<S>): T = visit(decl as KFuncDecl<S>)
    fun visit(decl: KBv2IntDecl): T = visit(decl as KFuncDecl<KIntSort>)
    fun <S : KBvSort> visit(decl: KBvAddNoOverflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvAddNoUnderflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvSubNoOverflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvSubNoUnderflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvDivNoOverflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvNegNoOverflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvMulNoOverflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
    fun <S : KBvSort> visit(decl: KBvMulNoUnderflowDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
}
