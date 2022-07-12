package org.ksmt.sort

import org.ksmt.KAst
import org.ksmt.KContext

abstract class KSort(ctx: KContext) : KAst(ctx) {
    abstract fun <T> accept(visitor: KSortVisitor<T>): T
}

class KBoolSort internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "Bool"
}

@Suppress("UnnecessaryAbstractClass")
abstract class KArithSort<out T : KArithSort<T>>(ctx: KContext) : KSort(ctx)

class KIntSort internal constructor(ctx: KContext) : KArithSort<KIntSort>(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "Int"
}

class KRealSort internal constructor(ctx: KContext) : KArithSort<KRealSort>(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "Real"
}

class KArraySort<out D : KSort, out R : KSort> internal constructor(
    ctx: KContext, val domain: D, val range: R
) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(): String = "(Array $domain $range)"
}

abstract class KBVSort(ctx: KContext) : KSort(ctx) {
    abstract val sizeBits: UInt

    override fun print(): String = "BitVec $sizeBits"
}

class KBV1Sort internal constructor(ctx: KContext) : KBVSort(ctx) {
    override val sizeBits: UInt = 1u

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBV8Sort internal constructor(ctx: KContext) : KBVSort(ctx) {
    override val sizeBits: UInt = Byte.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBV16Sort internal constructor(ctx: KContext) : KBVSort(ctx) {
    override val sizeBits: UInt = Short.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBV32Sort internal constructor(ctx: KContext) : KBVSort(ctx) {
    override val sizeBits: UInt = Int.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBV64Sort internal constructor(ctx: KContext) : KBVSort(ctx) {
    override val sizeBits: UInt = Long.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBVCustomSizeSort internal constructor(ctx: KContext, override val sizeBits: UInt) : KBVSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}
