package io.ksmt.sort

import io.ksmt.KAst
import io.ksmt.KContext
import io.ksmt.cache.hash

abstract class KSort(ctx: KContext) : KAst(ctx) {
    abstract fun <T> accept(visitor: KSortVisitor<T>): T
}

class KBoolSort internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("Bool")
    }

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean = this === other || other is KBoolSort
}

@Suppress("UnnecessaryAbstractClass")
abstract class KArithSort(ctx: KContext) : KSort(ctx)

class KIntSort internal constructor(ctx: KContext) : KArithSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("Int")
    }

    override fun hashCode(): Int = hash(javaClass)
    override fun equals(other: Any?): Boolean = this === other || other is KIntSort
}

class KRealSort internal constructor(ctx: KContext) : KArithSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("Real")
    }

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean = this === other || other is KRealSort
}

class KStringSort internal constructor(ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append("String")
    }

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean = this === other || other is KStringSort
}

sealed class KArraySortBase<R : KSort>(ctx: KContext) : KSort(ctx) {
    abstract val domainSorts: List<KSort>
    abstract val range: R

    override fun print(builder: StringBuilder): Unit = with(builder) {
        append("(Array ")
        domainSorts.forEach {
            append(it)
            append(' ')
        }
        range.print(this)
        append(')')
    }
}

class KArraySort<D : KSort, R : KSort> internal constructor(
    ctx: KContext, val domain: D, override val range: R
) : KArraySortBase<R>(ctx) {

    override val domainSorts: List<KSort>
        get() = listOf(domain)

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun hashCode(): Int = hash(javaClass, domain, range)

    override fun equals(other: Any?): Boolean =
        this === other || (other is KArraySort<*, *> && domain == other.domain && range == other.range)

    companion object {
        const val DOMAIN_SIZE = 1
    }
}

class KArray2Sort<D0 : KSort, D1 : KSort, R : KSort> internal constructor(
    ctx: KContext, val domain0: D0, val domain1: D1, override val range: R
) : KArraySortBase<R>(ctx) {

    override val domainSorts: List<KSort>
        get() = listOf(domain0, domain1)

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun hashCode(): Int = hash(javaClass, domain0, domain1, range)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KArray2Sort<*, *, *>) return false
        return domain0 == other.domain0 && domain1 == other.domain1 && range == other.range
    }

    companion object {
        const val DOMAIN_SIZE = 2
    }
}

class KArray3Sort<D0 : KSort, D1 : KSort, D2 : KSort, R : KSort> internal constructor(
    ctx: KContext, val domain0: D0, val domain1: D1, val domain2: D2, override val range: R
) : KArraySortBase<R>(ctx) {

    override val domainSorts: List<KSort>
        get() = listOf(domain0, domain1, domain2)

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun hashCode(): Int = hash(javaClass, domain0, domain1, domain2, range)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KArray3Sort<*, *, *, *>) return false
        return domain0 == other.domain0 && domain1 == other.domain1 && domain2 == other.domain2 && range == other.range
    }

    companion object {
        const val DOMAIN_SIZE = 3
    }
}

class KArrayNSort<R : KSort> internal constructor(
    ctx: KContext, override val domainSorts: List<KSort>, override val range: R
) : KArraySortBase<R>(ctx) {
    init {
        require(domainSorts.size > KArray3Sort.DOMAIN_SIZE) {
            "Use specialized Array with domain size ${domainSorts.size}"
        }
    }

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun hashCode(): Int = hash(javaClass, domainSorts, range)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KArrayNSort<*>) return false
        return domainSorts == other.domainSorts && range == other.range
    }
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
        this === other || (other is KBvSort && sizeBits == other.sizeBits)
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

open class KBvCustomSizeSort(ctx: KContext, override val sizeBits: UInt) : KBvSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)
}

open class KUninterpretedSort(val name: String, ctx: KContext) : KSort(ctx) {
    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun print(builder: StringBuilder) {
        builder.append(name)
    }

    override fun hashCode(): Int = hash(javaClass, name)

    override fun equals(other: Any?): Boolean =
        this === other || (other is KUninterpretedSort && name == other.name)
}

sealed class KFpSort(ctx: KContext, val exponentBits: UInt, val significandBits: UInt) : KSort(ctx) {
    override fun print(builder: StringBuilder) {
        builder.append("(FloatingPoint $exponentBits $significandBits)")
    }

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    abstract fun exponentShiftSize(): Int

    override fun hashCode(): Int = hash(javaClass, exponentBits, significandBits)

    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (other !is KFpSort) return false
        return exponentBits == other.exponentBits && significandBits == other.significandBits
    }
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

open class KFpCustomSizeSort(
    ctx: KContext,
    exponentBits: UInt,
    significandBits: UInt
) : KFpSort(ctx, exponentBits, significandBits) {
    override fun exponentShiftSize(): Int = (1 shl (exponentBits.toInt() - 1)) - 1
}

class KFpRoundingModeSort(ctx: KContext) : KSort(ctx) {
    override fun print(builder: StringBuilder) {
        builder.append("RoundingMode")
    }

    override fun <T> accept(visitor: KSortVisitor<T>): T = visitor.visit(this)

    override fun hashCode(): Int = hash(javaClass)

    override fun equals(other: Any?): Boolean =
        this === other || other is KFpRoundingModeSort
}
