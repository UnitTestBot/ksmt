package org.ksmt.sort

interface KSortVisitor<T> {
    fun visit(sort: KSort): Any = error("visitor is not implemented for sort $sort")
    fun visit(sort: KBoolSort): T
    fun visit(sort: KIntSort): T
    fun visit(sort: KRealSort): T
    fun <S : KBvSort> visit(sort: S): T
    fun <S : KFpSort> visit(sort: S): T
    fun <S: KFpRoundingModeSort> visit(sort: S): T
    fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): T
    fun visit(sort: KUninterpretedSort): T
}
