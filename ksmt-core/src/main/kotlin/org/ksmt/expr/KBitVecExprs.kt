package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort

abstract class KBitVecValue<S : KBvSort>(
    ctx: KContext
) : KApp<S, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>> = emptyList()
}

class KBitVec1Value internal constructor(ctx: KContext, val value: Boolean) : KBitVecValue<KBv1Sort>(ctx) {
    override fun accept(transformer: KTransformer): KExpr<KBv1Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv1Sort> = ctx.mkBvDecl(value)

    override fun sort(): KBv1Sort = ctx.mkBv1Sort()
}

abstract class KBitVecNumberValue<S : KBvSort, N : Number>(
    ctx: KContext,
    val numberValue: N
) : KBitVecValue<S>(ctx)

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
    val binaryStringValue: String,
    private val sizeBits: UInt
) : KBitVecValue<KBvSort>(ctx) {
    init {
        require(binaryStringValue.length.toUInt() == sizeBits) {
            "Provided string $binaryStringValue consist of ${binaryStringValue.length} " +
                    "symbols, but $sizeBits were expected"
        }
    }

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvDecl(binaryStringValue, sizeBits)

    override fun sort(): KBvSort = ctx.mkBvSort(sizeBits)
}

// expressions for operations
/**
 * Bitwise negation.
 */
class KBvNotExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<S> = ctx.mkBvNotDecl(value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Takes conjunction of bits in the [value], return a vector of length 1.
 */
class KBvReductionAndExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBv1Sort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBv1Sort> = ctx.mkBvReductionAndDecl(value.sort())

    override fun sort(): KBv1Sort = ctx.mkBv1Sort()

    override fun accept(transformer: KTransformer): KExpr<KBv1Sort> = transformer.transform(this)
}

/**
 * Take disjunction of bits in [value], return a vector of length 1.
 */
class KBvReductionOrExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBv1Sort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBv1Sort> = ctx.mkBvReductionOrDecl(value.sort())

    override fun sort(): KBv1Sort = ctx.mkBv1Sort()

    override fun accept(transformer: KTransformer): KExpr<KBv1Sort> = transformer.transform(this)
}

/**
 * Bitwise conjunction.
 */
class KBvAndExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvAndDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Bitwise disjunction.
 */
class KBvOrExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvOrDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Bitwise XOR.
 */
class KBvXorExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvXorDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Bitwise NAND.
 */
class KBvNAndExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvNAndDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Bitwise NOR.
 */
class KBvNorExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvNorDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Bitwise XNOR.
 */
class KBvXNorExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)


    override fun decl(): KDecl<S> = ctx.mkBvXNorDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Standard two's complement unary minus.
 */
class KBvNegationExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<S> = ctx.mkBvNegationDecl(value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Two's complement addition.
 */
class KBvAddExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvAddDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Two's complement subtraction.
 */
class KBvSubExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvSubDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Two's complement multiplication.
 */
class KBvMulExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvMulDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Unsigned division.
 *
 * It is defined as the floor of `arg0 / arg1` if `arg1` is different from zero.
 * Otherwise, the result is undefined.
 */
class KBvUnsignedDivExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvUnsignedDivDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Signed division.
 *
 * It is defined as:
 * * the floor of the `arg0 / arg1` if `arg1` is different from zero and `arg0 * arg1 >= 0`
 * * the ceiling of `arg0 / arg1` if `arg1` if different from zero and `arg0 * arg1 < 0`
 * * if `arg1` is zero, then the result is undefined.
 */
class KBvSignedDivExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvSignedDivDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Unsigned remainder.
 *
 * It is defined as `arg0 - (arg0 /u arg1) * arg1`, where `\u` represents unsigned division.
 * If `arg1` is zero, then the result is undefined.
 */
class KBvUnsignedRemExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvUnsignedRemDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Signed remainder.
 *
 * It is defined as `arg0 - (arg0 /s arg1) * arg1`, where `\s` represents signed division.
 * The most significant bit (sign) of the result is equal to the most significant bit of `arg0`.
 * If `arg1` is zero, then the result is undefined.
 */
class KBvSignedRemExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvSignedRemDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Two's complement signed remainder (sign follows divisor).
 * If `arg1` is zero, then the result is undefined.
 */
class KBvSignedModExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvSignedModDecl(arg0.sort(), arg1.sort())
    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Unsigned less-than.
 */
class KBvUnsignedLessExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedLessDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed less-than.
 */
class KBvSignedLessExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedLessDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Unsigned less-than or equal to.
 */
class KBvUnsignedLessOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedLessOrEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()


    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed less-than or equal to.
 */
class KBvSignedLessOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedLessOrEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()


    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Unsigned greater than or equal to.
 */
class KBvUnsignedGreaterOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedGreaterOrEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed greater than or equal to.
 */
class KBvSignedGreaterOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedGreaterOrEqualDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Unsigned greater-than.
 */
class KBvUnsignedGreaterExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUnsignedGreaterDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Two's complement signed greater-than.
 */
class KBvSignedGreaterExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSignedGreaterDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

/**
 * Bit-vector concatenation.
 *
 * @return a bit-vector of size `n1 + n2`, where `n1` and `n2` are the sizes of [arg0] and [arg1] correspondingly.
 */
class KBvConcatExpr internal constructor(
    ctx: KContext,
    val arg0: KExpr<KBvSort>,
    val arg1: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvConcatDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBvSort = decl().sort

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bit-vector extraction.
 *
 * Extracts the bits [high] down to [low] from the bitvector [value] of size `m`
 * to yield a new bitvector of size `n`, where `n = [high] - [low] + 1`.
 */
class KBvExtractExpr internal constructor(
    ctx: KContext,
    val high: Int,
    val low: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvExtractDecl(high, low, value)

    override fun sort(): KBvSort = ctx.mkBvSort((high - low + 1).toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bit-vector sign extension.
 *
 * Sign-extends the [value] to the (signed) equivalent bitvector of size `m + [i]`,
 * where `m` is the size of the [value].
 */
class KBvSignExtensionExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSignExtensionDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits + i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Bit-vector zero extension.
 *
 * Extend the [value] with zeros to the (unsigned) equivalent bitvector
 * of size `m + [i]`, where `m` is the size of the [value].
 */
class KBvZeroExtensionExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvZeroExtensionDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits + i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Bit-vector repetition.
 */
class KBvRepeatExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRepeatDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits * i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Shift left.
 *
 * It is equivalent to multiplication by `2^x`, where `x` is the value of [arg1].
 */
class KBvShiftLeftExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvShiftLeftDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)

}

/**
 * Logical shift right.
 *
 * It is equivalent to unsigned division by `2^x`, where `x` is the value of [arg1].
 */
class KBvLogicalShiftRightExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvLogicalShiftRightDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)

}

/**
 * Arithmetic shift right.
 *
 * It is like logical shift right except that the most significant bits
 * of the result always copy the most significant bit of the second argument.
 */
class KBvArithShiftRightExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvArithShiftRightDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)

}

/**
 * Rotate left.
 *
 * Rotates bits of the [arg0] to the left [arg1] times.
 */
class KBvRotateLeftExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvRotateLeftDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Rotate left.
 *
 * Rotates bits of the [value] to the left [i] times.
 */
class KBvRotateLeftIndexedExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<S> = ctx.mkBvRotateLeftIndexedDecl(i, value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Rotate right.
 *
 * Rotates bits of the [arg0] to the right [arg1] times.
 */
class KBvRotateRightExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<S> = ctx.mkBvRotateRightDecl(arg0.sort(), arg1.sort())

    override fun sort(): S = arg0.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
}

/**
 * Rotate right.
 *
 * Rotates bits of the [value] to the right [i] times.
 */
class KBvRotateRightIndexedExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<S>
) : KApp<S, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<S> = ctx.mkBvRotateRightIndexedDecl(i, value.sort())

    override fun sort(): S = value.sort()

    override fun accept(transformer: KTransformer): KExpr<S> = transformer.transform(this)
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
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override fun decl(): KDecl<KIntSort> = ctx.mkBv2IntDecl(value.sort(), isSigned)

    override fun sort(): KIntSort = ctx.mkIntSort()

    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)
}

class KBvAddNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val isSigned: Boolean
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvAddNoOverflowDecl(arg0.sort(), arg1.sort(), isSigned)

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvAddNoUnderflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvAddNoUnderflowDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSubNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSubNoOverflowDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSubNoUnderflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val isSigned: Boolean
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSubNoUnderflowDecl(arg0.sort(), arg1.sort(), isSigned)

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvDivNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvDivNoOverflowDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvNegNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>,
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvNegationNoOverflowDecl(value.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvMulNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val isSigned: Boolean
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvMulNoOverflowDecl(arg0.sort(), arg1.sort(), isSigned)

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvMulNoUnderflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, KExpr<S>>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvMulNoUnderflowDecl(arg0.sort(), arg1.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
