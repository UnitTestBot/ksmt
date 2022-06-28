package org.ksmt.sort

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.expr.KBVSize

abstract class KSort(ctx: KContext) : KAst(ctx) {
    abstract fun <T> accept(visitor: KSortVisitor<T>): T
}

class KBoolSort internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "Bool"
}

abstract class KArithSort<out T : KArithSort<T>>(ctx: KContext) : KSort(ctx)

class KIntSort internal constructor(ctx: KContext) : KArithSort<KIntSort>(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "Int"
}

class KRealSort internal constructor(ctx: KContext) : KArithSort<KRealSort>(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "Real"
}

class KBVSort<S : KBVSize> internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "BV"
}

class KArraySort<D : KSort, R : KSort> internal constructor(ctx: KContext, val domain: D, val range: R) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "(Array $domain $range)"
}
