package org.ksmt.sort

import org.ksmt.KAst
import org.ksmt.KContext

abstract class KSort(ctx: KContext) : KAst(ctx) {
    abstract fun <T> accept(visitor: KSortVisitor<T>): T
}

class KBoolSort internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(builder: StringBuilder) {
        builder.append("Bool")
    }
}

@Suppress("UnnecessaryAbstractClass")
abstract class KArithSort<out T : KArithSort<T>>(ctx: KContext) : KSort(ctx)

class KIntSort internal constructor(ctx: KContext) : KArithSort<KIntSort>(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(builder: StringBuilder) {
        builder.append("Int")
    }
}

class KRealSort internal constructor(ctx: KContext) : KArithSort<KRealSort>(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(builder: StringBuilder) {
        builder.append("Real")
    }
}

class KArraySort<out D : KSort, out R : KSort> internal constructor(
    ctx: KContext, val domain: D, val range: R
) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
    override fun print(builder: StringBuilder): Unit = with(builder) {
        append("(Array ")
        domain.print(this)
        append(' ')
        range.print(this)
        append(')')
    }
}

abstract class KBvSort(ctx: KContext) : KSort(ctx) {
    abstract val sizeBits: UInt

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append("(BitVec ")
        append(sizeBits)
        append(')')
    }
}

class KBv1Sort internal constructor(ctx: KContext) : KBvSort(ctx) {
    override val sizeBits: UInt = 1u

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBv8Sort internal constructor(ctx: KContext) : KBvSort(ctx) {
    override val sizeBits: UInt = Byte.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBv16Sort internal constructor(ctx: KContext) : KBvSort(ctx) {
    override val sizeBits: UInt = Short.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBv32Sort internal constructor(ctx: KContext) : KBvSort(ctx) {
    override val sizeBits: UInt = Int.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBv64Sort internal constructor(ctx: KContext) : KBvSort(ctx) {
    override val sizeBits: UInt = Long.SIZE_BITS.toUInt()

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KBvCustomSizeSort internal constructor(ctx: KContext, override val sizeBits: UInt) : KBvSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

class KUninterpretedSort internal constructor(val name: String, ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append(name)
    }
}
