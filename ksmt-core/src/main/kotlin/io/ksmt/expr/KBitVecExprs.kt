package io.ksmt.expr

import io.ksmt.KContext
import io.ksmt.cache.hash
import io.ksmt.cache.structurallyEqual
import io.ksmt.decl.KDecl
import io.ksmt.expr.transformer.KTransformerBase
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KIntSort
import io.ksmt.utils.toBinary
import java.math.BigInteger

abstract class KBitVecValue<S : KBvSort>(ctx: KContext) : KInterpretedValue<S>(ctx){
    abstract val stringValue: String
}

class KBitVec1Value internal constructor(
    ctx: KContext,
    @JvmField val value: Boolean
) : KBitVecValue<KBv1Sort>(ctx) {
    override fun accept(transformer: KTransformerBase): KExpr<KBv1Sort> = transformer.transform(this)

    override val stringValue: String
        get() = if (value) "1" else "0"

    override val decl: KDecl<KBv1Sort>
        get() = ctx.mkBvDecl(value)

    override val sort: KBv1Sort = ctx.bv1Sort

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

abstract class KBitVecNumberValue<S : KBvSort, N : Number>(ctx: KContext) : KBitVecValue<S>(ctx) {
    abstract val numberValue: N

    override val stringValue: String
        get() = numberValue.toBinary()

    override fun internHashCode(): Int = hash(numberValue)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { numberValue }
}

class KBitVec8Value internal constructor(
    ctx: KContext,
    @JvmField val byteValue: Byte
) : KBitVecNumberValue<KBv8Sort, Byte>(ctx) {
    override fun accept(transformer: KTransformerBase): KExpr<KBv8Sort> = transformer.transform(this)

    override val numberValue: Byte
        get() = byteValue

    override val decl: KDecl<KBv8Sort>
        get() = ctx.mkBvDecl(byteValue)

    override val sort: KBv8Sort = ctx.bv8Sort

    override fun internHashCode(): Int = hash(byteValue)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { byteValue }
}

class KBitVec16Value internal constructor(
    ctx: KContext,
    @JvmField val shortValue: Short
) : KBitVecNumberValue<KBv16Sort, Short>(ctx) {
    override fun accept(transformer: KTransformerBase): KExpr<KBv16Sort> = transformer.transform(this)

    override val numberValue: Short
        get() = shortValue

    override val decl: KDecl<KBv16Sort>
        get() = ctx.mkBvDecl(shortValue)

    override val sort: KBv16Sort = ctx.bv16Sort

    override fun internHashCode(): Int = hash(shortValue)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { shortValue }
}

class KBitVec32Value internal constructor(
    ctx: KContext,
    @JvmField val intValue: Int
) : KBitVecNumberValue<KBv32Sort, Int>(ctx) {
    override fun accept(transformer: KTransformerBase): KExpr<KBv32Sort> = transformer.transform(this)

    override val numberValue: Int
        get() = intValue

    override val decl: KDecl<KBv32Sort>
        get() = ctx.mkBvDecl(intValue)

    override val sort: KBv32Sort = ctx.bv32Sort

    override fun internHashCode(): Int = hash(intValue)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { intValue }
}

class KBitVec64Value internal constructor(
    ctx: KContext,
    @JvmField val longValue: Long
) : KBitVecNumberValue<KBv64Sort, Long>(ctx) {
    override fun accept(transformer: KTransformerBase): KExpr<KBv64Sort> = transformer.transform(this)

    override val numberValue: Long
        get() = longValue

    override val decl: KDecl<KBv64Sort>
        get() = ctx.mkBvDecl(longValue)

    override val sort: KBv64Sort = ctx.bv64Sort

    override fun internHashCode(): Int = hash(longValue)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { longValue }
}

class KBitVecCustomValue internal constructor(
    ctx: KContext,
    val value: BigInteger,
    val sizeBits: UInt
) : KBitVecValue<KBvSort>(ctx) {

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override val stringValue: String
        get() = value.toBinary(sizeBits)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkBvDecl(value, sizeBits)

    override val sort: KBvSort = ctx.mkBvSort(sizeBits)

    override fun internHashCode(): Int = hash(sizeBits, value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { sizeBits }, { value })
}

// expressions for operations
/**
 * Bitwise negation.
 */
class KBvNotExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<S>
        get() = ctx.mkBvNotDecl(sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = value.sort

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

/**
 * Takes conjunction of bits in the [value], return a vector of length 1.
 */
class KBvReductionAndExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBv1Sort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBv1Sort>
        get() = ctx.mkBvReductionAndDecl(value.sort)

    override val sort: KBv1Sort = ctx.bv1Sort

    override fun accept(transformer: KTransformerBase): KExpr<KBv1Sort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

/**
 * Take disjunction of bits in [value], return a vector of length 1.
 */
class KBvReductionOrExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBv1Sort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBv1Sort>
        get() = ctx.mkBvReductionOrDecl(value.sort)

    override val sort: KBv1Sort = ctx.bv1Sort

    override fun accept(transformer: KTransformerBase): KExpr<KBv1Sort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

/**
 * Bitwise conjunction.
 */
class KBvAndExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvAndDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Bitwise disjunction.
 */
class KBvOrExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvOrDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Bitwise XOR.
 */
class KBvXorExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvXorDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Bitwise NAND.
 */
class KBvNAndExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvNAndDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Bitwise NOR.
 */
class KBvNorExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvNorDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Bitwise XNOR.
 */
class KBvXNorExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvXNorDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Standard two's complement unary minus.
 */
class KBvNegationExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<S>
        get() = ctx.mkBvNegationDecl(sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = value.sort

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

/**
 * Two's complement addition.
 */
class KBvAddExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvAddDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement subtraction.
 */
class KBvSubExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvSubDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement multiplication.
 */
class KBvMulExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvMulDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
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
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvUnsignedDivDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
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
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvSignedDivDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
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
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvUnsignedRemDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
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
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvSignedRemDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement signed remainder (sign follows divisor).
 * If `arg1` is zero, then the result is undefined.
 */
class KBvSignedModExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkBvSignedModDecl(sort, sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg0.sort

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Unsigned less-than.
 */
class KBvUnsignedLessExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvUnsignedLessDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement signed less-than.
 */
class KBvSignedLessExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvSignedLessDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Unsigned less-than or equal to.
 */
class KBvUnsignedLessOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvUnsignedLessOrEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement signed less-than or equal to.
 */
class KBvSignedLessOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvSignedLessOrEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Unsigned greater than or equal to.
 */
class KBvUnsignedGreaterOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvUnsignedGreaterOrEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement signed greater than or equal to.
 */
class KBvSignedGreaterOrEqualExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvSignedGreaterOrEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Unsigned greater-than.
 */
class KBvUnsignedGreaterExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvUnsignedGreaterDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

/**
 * Two's complement signed greater-than.
 */
class KBvSignedGreaterExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvSignedGreaterDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
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
) : KApp<KBvSort, KBvSort>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkBvConcatDecl(arg0.sort, arg1.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override val sort: KBvSort = ctx.mkBvSort(arg0.sort.sizeBits + arg1.sort.sizeBits)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
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
) : KApp<KBvSort, KBvSort>(ctx) {
    init {
        require(low <= high) { "High bit $high must be greater than lower bit $low" }
    }

    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkBvExtractDecl(high, low, value)

    override val sort: KBvSort = ctx.mkBvSort((high - low + 1).toUInt())

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value, high, low)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { high }, { low })
}

/**
 * Bit-vector sign extension.
 *
 * Sign-extends the [value] to the (signed) equivalent bitvector of size `m + [extensionSize]`,
 * where `m` is the size of the [value].
 */
class KBvSignExtensionExpr internal constructor(
    ctx: KContext,
    val extensionSize: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KBvSort>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkBvSignExtensionDecl(extensionSize, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override val sort: KBvSort = ctx.mkBvSort(value.sort.sizeBits + extensionSize.toUInt())

    override fun internHashCode(): Int = hash(value, extensionSize)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { extensionSize })
}

/**
 * Bit-vector zero extension.
 *
 * Extend the [value] with zeros to the (unsigned) equivalent bitvector
 * of size `m + [extensionSize]`, where `m` is the size of the [value].
 */
class KBvZeroExtensionExpr internal constructor(
    ctx: KContext,
    val extensionSize: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KBvSort>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkBvZeroExtensionDecl(extensionSize, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override val sort: KBvSort = ctx.mkBvSort(value.sort.sizeBits + extensionSize.toUInt())

    override fun internHashCode(): Int = hash(value, extensionSize)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { extensionSize })
}

/**
 * Bit-vector repetition.
 */
class KBvRepeatExpr internal constructor(
    ctx: KContext,
    val repeatNumber: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KBvSort>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkBvRepeatDecl(repeatNumber, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override val sort: KBvSort = ctx.mkBvSort(value.sort.sizeBits * repeatNumber.toUInt())

    override fun internHashCode(): Int = hash(value, repeatNumber)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { repeatNumber })
}

/**
 * Shift left.
 *
 * It is equivalent to multiplication by `2^x`, where `x` is the value of [shift].
 */
class KBvShiftLeftExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg: KExpr<S>,
    val shift: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg, shift)

    override val decl: KDecl<S>
        get() = ctx.mkBvShiftLeftDecl(arg.sort, shift.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg.sort

    override fun internHashCode(): Int = hash(arg, shift)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { shift })
}

/**
 * Logical shift right.
 *
 * It is equivalent to unsigned division by `2^x`, where `x` is the value of [shift].
 */
class KBvLogicalShiftRightExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg: KExpr<S>,
    val shift: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg, shift)

    override val decl: KDecl<S>
        get() = ctx.mkBvLogicalShiftRightDecl(arg.sort, shift.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg.sort

    override fun internHashCode(): Int = hash(arg, shift)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { shift })
}

/**
 * Arithmetic shift right.
 *
 * It is like logical shift right except that the most significant bits
 * of the result always copy the most significant bit of the second argument.
 */
class KBvArithShiftRightExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg: KExpr<S>,
    val shift: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg, shift)

    override val decl: KDecl<S>
        get() = ctx.mkBvArithShiftRightDecl(arg.sort, shift.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg.sort

    override fun internHashCode(): Int = hash(arg, shift)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { shift })
}

/**
 * Rotate left.
 *
 * Rotates bits of the [arg] to the left [rotation] times.
 */
class KBvRotateLeftExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg: KExpr<S>,
    val rotation: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg, rotation)

    override val decl: KDecl<S>
        get() = ctx.mkBvRotateLeftDecl(arg.sort, rotation.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg.sort

    override fun internHashCode(): Int = hash(arg, rotation)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { rotation })
}

/**
 * Rotate left.
 *
 * Rotates bits of the [value] to the left [rotationNumber] times.
 */
class KBvRotateLeftIndexedExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val rotationNumber: Int,
    val value: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<S>
        get() = ctx.mkBvRotateLeftIndexedDecl(rotationNumber, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = value.sort

    override fun internHashCode(): Int = hash(value, rotationNumber)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { rotationNumber })
}

/**
 * Rotate right.
 *
 * Rotates bits of the [arg] to the right [rotation] times.
 */
class KBvRotateRightExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg: KExpr<S>,
    val rotation: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg, rotation)

    override val decl: KDecl<S>
        get() = ctx.mkBvRotateRightDecl(arg.sort, rotation.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = arg.sort

    override fun internHashCode(): Int = hash(arg, rotation)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg }, { rotation })
}

/**
 * Rotate right.
 *
 * Rotates bits of the [value] to the right [rotationNumber] times.
 */
class KBvRotateRightIndexedExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val rotationNumber: Int,
    val value: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<S>
        get() = ctx.mkBvRotateRightIndexedDecl(rotationNumber, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S = value.sort

    override fun internHashCode(): Int = hash(value, rotationNumber)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { rotationNumber })
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
) : KApp<KIntSort, KBvSort>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(value)

    override val decl: KDecl<KIntSort>
        get() = ctx.mkBv2IntDecl(value.sort, isSigned)

    override val sort: KIntSort = ctx.intSort

    override fun accept(transformer: KTransformerBase): KExpr<KIntSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value, isSigned)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { value }, { isSigned })
}

class KBvAddNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val isSigned: Boolean
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvAddNoOverflowDecl(arg0.sort, arg1.sort, isSigned)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, isSigned)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { isSigned })
}

class KBvAddNoUnderflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvAddNoUnderflowDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KBvSubNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvSubNoOverflowDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KBvSubNoUnderflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val isSigned: Boolean
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvSubNoUnderflowDecl(arg0.sort, arg1.sort, isSigned)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, isSigned)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { isSigned })
}

class KBvDivNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvDivNoOverflowDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}

class KBvNegNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>,
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvNegationNoOverflowDecl(value.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(value)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other) { value }
}

class KBvMulNoOverflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val isSigned: Boolean
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvMulNoOverflowDecl(arg0.sort, arg1.sort, isSigned)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1, isSigned)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 }, { isSigned })
}

class KBvMulNoUnderflowExpr<S : KBvSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkBvMulNoUnderflowDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun internHashCode(): Int = hash(arg0, arg1)
    override fun internEquals(other: Any): Boolean = structurallyEqual(other, { arg0 }, { arg1 })
}
