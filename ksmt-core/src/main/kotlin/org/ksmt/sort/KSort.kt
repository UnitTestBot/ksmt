package org.ksmt.sort

import org.ksmt.KAst
import org.ksmt.KContext
import org.ksmt.cache.hash

abstract class KSort(ctx: KContext) : KAst(ctx) {
    abstract fun <T> accept(visitor: KSortVisitor<T>): T
}

class KBoolSort internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("Bool")
    }

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean = other is KBoolSort
}

@Suppress("UnnecessaryAbstractClass")
abstract class KArithSort(ctx: KContext) : KSort(ctx)

class KIntSort internal constructor(ctx: KContext) : KArithSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("Int")
    }

    override fun hashCode(): Int = hash(javaClass)
    override fun equals(other: Any?): Boolean = other is KIntSort
}

class KRealSort internal constructor(ctx: KContext) : KArithSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("Real")
    }

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean = other is KRealSort
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

    override fun hashCode(): Int = hash(javaClass, domain, range)

    override fun equals(other: Any?): Boolean =
        other is KArraySort<*, *> && domain == other.domain && range == other.range
}

abstract class KBvSort(ctx: KContext) : KSort(ctx) {
    abstract val sizeBits: UInt

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append("(BitVec ")
        append(sizeBits)
        append(')')
    }

    override fun hashCode(): Int = hash(javaClass, sizeBits)

    override fun equals(other: Any?): Boolean =
        other is KBvSort && sizeBits == other.sizeBits
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

    override fun hashCode(): Int = hash(javaClass, name)

    override fun equals(other: Any?): Boolean = other is KUninterpretedSort && name == other.name
}

sealed class KFpSort(ctx: KContext, val exponentBits: UInt, val significandBits: UInt) : KSort(ctx) {
    override fun print(builder: StringBuilder) {
        builder.append("FP (eBits: $exponentBits) (sBits: $significandBits)")
    }

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    abstract fun exponentShiftSize(): Int

    override fun hashCode(): Int = hash(javaClass, exponentBits, significandBits)

    override fun equals(other: Any?): Boolean =
        other is KFpSort && exponentBits == other.exponentBits && significandBits == other.significandBits
}

class KFp16Sort(ctx: KContext) : KFpSort(ctx, exponentBits, significandBits) {
    override fun exponentShiftSize(): Int = exponentShiftSize

    companion object {
        val exponentBits: UInt = 5.toUInt()
        val significandBits: UInt = 11.toUInt()
        const val exponentShiftSize: Int = 15
    }
}

class KFp32Sort(ctx: KContext) : KFpSort(ctx, exponentBits, significandBits) {
    override fun exponentShiftSize(): Int = exponentShiftSize

    companion object {
        val exponentBits: UInt = 8.toUInt()
        val significandBits: UInt = 24.toUInt()
        const val exponentShiftSize: Int = 127
    }
}

class KFp64Sort(ctx: KContext) : KFpSort(ctx, exponentBits, significandBits) {
    override fun exponentShiftSize(): Int = exponentShiftSize

    companion object {
        val exponentBits: UInt = 11.toUInt()
        val significandBits: UInt = 53.toUInt()
        const val exponentShiftSize: Int = 1023
    }
}

class KFp128Sort(ctx: KContext) : KFpSort(ctx, exponentBits, significandBits) {
    override fun exponentShiftSize(): Int = exponentShiftSize

    companion object {
        val exponentBits: UInt = 15.toUInt()
        val significandBits: UInt = 113.toUInt()
        const val exponentShiftSize: Int = 16383
    }
}

class KFpCustomSizeSort(
    ctx: KContext,
    exponentBits: UInt,
    significandBits: UInt
) : KFpSort(ctx, exponentBits, significandBits) {
    override fun exponentShiftSize(): Int = (1 shl (exponentBits.toInt() - 1)) - 1
}

class KFpRoundingModeSort(ctx: KContext) : KSort(ctx) {
    override fun print(builder: StringBuilder) {
        builder.append("RoundingModeSort")
    }

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean = other is KFpRoundingModeSort
}
