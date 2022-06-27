package org.ksmt.decl

import org.ksmt.sort.KArithSort
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KBoolSort
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
    fun visit(decl: KFalseDecl): T = visit(decl as KConstDecl<KBoolSort>)
    fun visit(decl: KTrueDecl): T = visit(decl as KConstDecl<KBoolSort>)
    fun <S : KSort> visit(decl: KEqDecl<S>): T = visit(decl as KFuncDecl<KBoolSort>)
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
}
