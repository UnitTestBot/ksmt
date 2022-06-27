package org.ksmt.sort

import org.ksmt.expr.KBVSize

interface KSortVisitor<T> {
    fun visit(sort: KSort): Any = error("visitor is not implemented for sort $sort")
    fun visit(sort: KBoolSort): T
    fun visit(sort: KIntSort): T
    fun visit(sort: KRealSort): T
    fun <S : KBVSize> visit(sort: KBVSort<S>): T
    fun <D : KSort, R : KSort> visit(sort: KArraySort<D, R>): T
}
