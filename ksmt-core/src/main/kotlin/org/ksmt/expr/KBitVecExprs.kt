package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort

abstract class KBitVecValue<T : KBvSort>(
    ctx: KContext
) : KApp<T, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>> = emptyList()
}

class KBitVec1Value internal constructor(ctx: KContext, val value: Boolean) : KBitVecValue<KBv1Sort>(ctx) {
    override fun accept(transformer: KTransformer): KExpr<KBv1Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv1Sort> = ctx.mkBvDecl(value)

    override fun sort(): KBv1Sort = ctx.mkBv1Sort()
}

abstract class KBitVecNumberValue<T : KBvSort, N : Number>(
    ctx: KContext,
    val numberValue: N
) : KBitVecValue<T>(ctx)

class KBitVec8Value internal constructor(ctx: KContext, byteValue: Byte) :
    KBitVecNumberValue<KBv8Sort, Byte>(ctx, byteValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv8Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv8Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBv8Sort = ctx.mkBv8Sort()
}

class KBitVec16Value internal constructor(ctx: KContext, shortValue: Short) :
    KBitVecNumberValue<KBv16Sort, Short>(ctx, shortValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv16Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv16Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBv16Sort = ctx.mkBv16Sort()
}

class KBitVec32Value internal constructor(ctx: KContext, intValue: Int) :
    KBitVecNumberValue<KBv32Sort, Int>(ctx, intValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv32Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv32Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBv32Sort = ctx.mkBv32Sort()
}

class KBitVec64Value internal constructor(ctx: KContext, longValue: Long) :
    KBitVecNumberValue<KBv64Sort, Long>(ctx, longValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv64Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv64Sort> = ctx.mkBvDecl(numberValue)
    override fun sort(): KBv64Sort = ctx.mkBv64Sort()
}

class KBitVecCustomValue internal constructor(
    ctx: KContext,
    val value: String,
    private val sizeBits: UInt
) : KBitVecValue<KBvSort>(ctx) {

    init {
        require(value.length.toUInt() == sizeBits) {
            "Provided string $value consist of ${value.length} symbols, but $sizeBits were expected"
        }
    }

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvDecl(value, sizeBits)

    override fun sort(): KBvSort = ctx.mkBvSort(sizeBits)
}

// expressions for operations
/**
 * Bitwise negation.
 */
class KBvNotExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNotDecl(value.sort())

    override fun sort(): KBvSort = value.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Takes conjunction of bits in a vector, return vector of length 1.
 */
class KBvReductionAndExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvReductionAndDecl(value.sort())

    override fun sort(): KBvSort = ctx.mkBv1Sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Take disjunction of bits in a vector, return vector of length 1.
 */
class KBvReductionOrExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvReductionOrDecl(value.sort())

    override fun sort(): KBvSort = ctx.mkBv1Sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise conjunction.
 */
class KBvAndExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvAndDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise disjunction.
 */
class KBvOrExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvOrDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()
    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise XOR.
 */
class KBvXorExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvXorDecl(left.sort(), right.sort())
    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise NAND.
 */
class KBvNAndExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNAndDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise NOR.
 */
class KBvNorExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNorDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise XNOR.
 */
class KBvXNorExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvXNorDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Standard two's complement unary minus.
 */
class KBvNegationExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNegationDecl(value.sort())
    override fun sort(): KBvSort = value.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement addition.
 */
class KBvAddExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvAddDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement subtraction.
 */
class KBvSubExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSubDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement multiplication.
 */
class KBvMulExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvMulDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Unsigned division.
 *
 * It is defined as the floor of `t1 / t2` if `t2` is different from zero.
 * Otherwise, the result is undefined.
 */
class KBvUnsignedDivExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvUnsignedDivDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Signed division.
 *
 * It is defined as:
 * * the floor of the `t1 / t2` if `t2` is different from zero and `t1 * t2 >= 0`
 * * the ceiling of `t1 / t2` if `t2` if different from zero and `t1 * t2 < 0`
 * * if `t2` is zero, then the result is undefined.
 */
class KBvSignedDivExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSignedDivDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Unsigned remainder.
 *
 * It is defined as `t1 - (t1 /u t2) * t2`, where `\u` represents unsigned division.
 * If `t2` is zero, then the result is undefined.
 */
class KBvUnsignedRemExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvUnsignedRemDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Signed remainder.
 *
 * It is defined as `t1 - (t1 /s t2) * t2`, where `\s` represents signed division.
 * The most significant bit (sign) of the result is equal to the most significant bit of `t1`.
 * If `t2` is zero, then the result is undefined.
 */
class KBvSignedRemExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSignedRemDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement signed remainder (sign follows divisor).
 * If `t2` is zero, then the result is undefined.
 */
class KBvSignedModExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSignedModDecl(left.sort(), right.sort())
    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Unsigned less-than.
 */
class KBvUnsignedLessExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedLessDecl(left.sort(), right.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed less-than.
 */
class KBvSignedLessExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedLessDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Unsigned less-than or equal to.
 */
class KBvUnsignedLessOrEqualExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedLessOrEqualDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()


    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed less-than or equal to.
 */
class KBvSignedLessOrEqualExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedLessOrEqualDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()


    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Unsigned greater than or equal to.
 */
class KBvUnsignedGreaterOrEqualExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedGreaterOrEqualDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed greater than or equal to.
 */
class KBvSignedGreaterOrEqualExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedGreaterOrEqualDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Unsigned greater-than.
 */
class KBvUnsignedGreaterExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedGreaterDecl(left.sort(), right.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed greater-than.
 */
class KBvSignedGreaterExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedGreaterDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Bit-vector concatenation.
 *
 * @return a bit-vector of size `n1 + n2`, where `n1` and `n2` are the sizes of [left] and [right] correspondingly.
 */
class KConcatExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkConcatDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = decl().sort
    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bit-vector extraction.
 *
 * Extracts the bits [high] down to [low] from the bitvector [value] of size `m`
 * to yield a new bitvector of size `n`, where `n = [high] - [low] + 1`.
 */
class KExtractExpr internal constructor(
    ctx: KContext,
    val high: Int,
    val low: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkExtractDecl(high, low, value)

    override fun sort(): KBvSort = ctx.mkBvSort((high - low + 1).toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = TODO()
}

/**
 * Bit-vector sign extension.
 *
 * Sign-extends the [value] to the (signed) equivalent bitvector of size `m + [i]`,
 * where `m` is the size of the [value].
 */
class KSignExtensionExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkSignExtensionDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits + i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Bit-vector zero extension.
 *
 * Extend the [value] with zeros to the (unsigned) equivalent bitvector
 * of size `m + [i]`, where `m` is the size of the [value].
 */
class KZeroExtensionExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkZeroExtensionDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits + i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Bit-vector repetition.
 */
class KRepeatExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkRepeatDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits * i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Shift left.
 *
 * It is equivalent to multiplication by `2^x`, where `x` is the value of [right].
 */
class KBvShiftLeftExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvShiftLeftDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Logical shift right.
 *
 * It is equivalent to unsigned division by `2^x`, where `x` is the value of [right].
 */
class KBvLogicalShiftRightExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvLogicalShiftRightDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Arithmetic shift right.
 *
 * It is like logical shift right except that the most significant bits
 * of the result always copy the most significant bit of the second argument.
 */
class KBvArithShiftRightExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvArithShiftRightDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Rotate left.
 *
 * Rotates bits of the [left] to the left [right] times.
 */
class KBvRotateLeftExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRotateLeftDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Rotate left.
 *
 * Rotates bits of the [left] to the right [right] times.
 */
class KBvRotateRightExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRotateRightDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Creates an integer from the bit-vector argument [value].
 *
 * If [isSigned] is false, then the bit-vector [value] is treated as unsigned.
 * So the reuslt is non-negative and in the range `[0..2^(n - 1)]`,
 * where N are the number of bits in [value].
 *
 * If [isSigned] is true, then [value] is treated as a signed bit-vector.
 */
class KBv2IntExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>,
    val isSigned: Boolean
) : KApp<KIntSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KIntSort> = ctx.mkBv2IntDecl(value.sort(), isSigned)

    override fun sort(): KIntSort = ctx.mkIntSort()

    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

// TODO think about their declarations -- in z3 they have decl: declare-fun => (Bool Bool) Bool
// toString for `mkBvAddNoOverflow(mkBv(2, 2), mkBv(2, 2), true)`
// (=> (and (bvslt #b00 #b10) (bvslt #b00 #b10)) (bvslt #b00 (bvadd #b10 #b10)))
class KBvAddNoOverflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
    val isSigned: Boolean
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvAddNoOverflowDecl(left.sort(), right.sort(), isSigned)
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvAddNoUnderflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvAddNoUnderflowDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSubNoOverflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSubNoOverflowDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSubNoUnderflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
    val isSigned: Boolean
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSubNoUnderflowDecl(left.sort(), right.sort(), isSigned)
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvDivNoOverflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvDivNoOverflowDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvNegNoOverflowExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>,
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvNegNoOverflowDecl(value.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvMulNoOverflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
    val isSigned: Boolean
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvMulNoOverflowDecl(left.sort(), right.sort(), isSigned)
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvMulNoUnderflowExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>,
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvMulNoUnderflowDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}