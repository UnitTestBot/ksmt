package org.ksmt.sort

import org.ksmt.expr.KBVSize

interface KSort {
    fun <T> accept(visitor: KSortVisitor<T>): T
}

object KBoolSort : KSort {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

interface KArithSort<T : KArithSort<T>> : KSort

object KIntSort : KArithSort<KIntSort> {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

object KRealSort : KArithSort<KRealSort> {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBVSort<S : KBVSize> : KSort {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KArraySort<D : KSort, R : KSort>(val domain: D, val range: R) : KSort {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}
