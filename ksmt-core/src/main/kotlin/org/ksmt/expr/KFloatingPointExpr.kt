package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.cache.hash
import org.ksmt.cache.structurallyEqual
import org.ksmt.decl.KDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.sort.KSort
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.getExponent
import org.ksmt.utils.getHalfPrecisionExponent
import org.ksmt.utils.halfPrecisionSignificand
import org.ksmt.utils.significand
import org.ksmt.utils.uncheckedCast

abstract class KFpValue<T : KFpSort>(ctx: KContext) : KInterpretedValue<T>(ctx) {
    abstract val significand: KBitVecValue<out KBvSort>
    abstract val biasedExponent: KBitVecValue<out KBvSort>
    abstract val signBit: Boolean
}

/**
 * Fp16 value. Note that [value] should has biased Fp32 exponent,
 * but a constructed Fp16 will have an unbiased one.
 *
 * Fp32 to Fp16 transformation:
 * sign   exponent         significand
 * 0      00000000   00000000000000000000000    (1 8 23)
 * x      x___xxxx   xxxxxxxxxx_____________    (1 5 10)
 */
class KFp16Value internal constructor(
    ctx: KContext,
    val value: Float
) : KFpValue<KFp16Sort>(ctx) {

    override val significand by lazy {
        with(ctx) {
            value.halfPrecisionSignificand.toBv(KFp16Sort.significandBits - 1u)
        }
    }

    override val biasedExponent by lazy {
        with(ctx) {
            value.getHalfPrecisionExponent(isBiased = true).toBv(KFp16Sort.exponentBits)
        }
    }

    override val signBit by lazy {
        value.booleanSignBit
    }

    override val decl: KDecl<KFp16Sort>
        get() = ctx.mkFp16Decl(value)

    override val sort: KFp16Sort
        get() = ctx.mkFp16Sort()

    override fun accept(transformer: KTransformerBase): KExpr<KFp16Sort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFp32Value internal constructor(
    ctx: KContext,
    val value: Float
) : KFpValue<KFp32Sort>(ctx) {

    override val significand by lazy {
        with(ctx) {
            value.significand.toBv(KFp32Sort.significandBits - 1u)
        }
    }

    override val biasedExponent by lazy {
        with(ctx) {
            value.getExponent(isBiased = true).toBv(KFp32Sort.exponentBits)
        }
    }

    override val signBit by lazy {
        value.booleanSignBit
    }

    override val decl: KDecl<KFp32Sort>
        get() = ctx.mkFp32Decl(value)

    override val sort: KFp32Sort
        get() = ctx.mkFp32Sort()

    override fun accept(transformer: KTransformerBase): KExpr<KFp32Sort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFp64Value internal constructor(
    ctx: KContext,
    val value: Double
) : KFpValue<KFp64Sort>(ctx) {

    override val significand by lazy {
        with(ctx) {
            value.significand.toBv(KFp64Sort.significandBits - 1u)
        }
    }

    override val biasedExponent by lazy {
        with(ctx) {
            value.getExponent(isBiased = true).toBv(KFp64Sort.exponentBits)
        }
    }

    override val signBit by lazy {
        value.booleanSignBit
    }

    override val decl: KDecl<KFp64Sort>
        get() = ctx.mkFp64Decl(value)

    override val sort: KFp64Sort
        get() = ctx.mkFp64Sort()

    override fun accept(transformer: KTransformerBase): KExpr<KFp64Sort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

/**
 * KFp128 value.
 *
 * Note: if [biasedExponent] contains more than [KFp128Sort.exponentBits] meaningful bits,
 * only the last [KFp128Sort.exponentBits] of then will be taken.
 */
class KFp128Value internal constructor(
    ctx: KContext,
    override val significand: KBitVecValue<*>,
    override val biasedExponent: KBitVecValue<*>,
    override val signBit: Boolean
) : KFpValue<KFp128Sort>(ctx) {
    init {
        require(biasedExponent.sort.sizeBits == KFp128Sort.exponentBits) {
            "Exponent size must be ${KFp128Sort.exponentBits}."
        }
        require(significand.sort.sizeBits == KFp128Sort.significandBits - 1u) {
            "Significand size must be ${KFp128Sort.significandBits - 1u}."
        }
    }

    override val decl: KDecl<KFp128Sort>
        get() = ctx.mkFp128DeclBiased(
            significandBits = significand,
            biasedExponent = biasedExponent,
            signBit = signBit
        )

    override val sort: KFp128Sort
        get() = ctx.mkFp128Sort()

    override fun accept(transformer: KTransformerBase): KExpr<KFp128Sort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(signBit, biasedExponent, significand)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { signBit }, { biasedExponent }, { significand })
}

/**
 * KFp value of custom size.
 */
class KFpCustomSizeValue internal constructor(
    ctx: KContext,
    val significandSize: UInt,
    val exponentSize: UInt,
    override val significand: KBitVecValue<*>,
    override val biasedExponent: KBitVecValue<*>,
    override val signBit: Boolean
) : KFpValue<KFpSort>(ctx) {
    init {
        require(biasedExponent.sort.sizeBits == exponentSize) {
            "Exponent size must be $exponentSize."
        }
        require(significand.sort.sizeBits == significandSize - 1u) {
            "Significand size must be ${significandSize - 1u}."
        }
    }

    override val decl: KDecl<KFpSort>
        get() = ctx.mkFpCustomSizeDeclBiased(
            significandSize = significandSize,
            exponentSize = exponentSize,
            significand = significand,
            biasedExponent = biasedExponent,
            signBit = signBit
        )

    override val sort: KFpSort
        get() = ctx.mkFpSort(exponentSize, significandSize)

    override fun accept(transformer: KTransformerBase): KExpr<KFpSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(signBit, biasedExponent, significand)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { signBit }, { biasedExponent }, { significand })
}

class KFpAbsExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<S>
        get() = ctx.mkFpAbsDecl(value.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = value.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += value
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

/**
 * Inverts the sign bit.
 */
class KFpNegationExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<S>
        get() = ctx.mkFpNegationDecl(value.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = value.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += value
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

// TODO Can they have different sorts?
class KFpAddExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, arg0, arg1).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpAddDecl(roundingMode.sort, arg0.sort, arg1.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { arg0 }, { arg1 })
}

class KFpSubExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, arg0, arg1).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpSubDecl(roundingMode.sort, arg0.sort, arg1.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { arg0 }, { arg1 })
}

class KFpMulExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, arg0, arg1).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpMulDecl(roundingMode.sort, arg0.sort, arg1.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { arg0 }, { arg1 })
}

class KFpDivExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, arg0, arg1).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpDivDecl(roundingMode.sort, arg0.sort, arg1.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { arg0 }, { arg1 })
}

class KFpFusedMulAddExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>,
    val arg2: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, arg0, arg1, arg2).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpFusedMulAddDecl(
            roundingMode.sort,
            arg0.sort,
            arg1.sort,
            arg2.sort
        )


    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, arg0, arg1, arg2)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { arg0 }, { arg1 }, { arg2 })

}

class KFpSqrtExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val value: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, value).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpSqrtDecl(roundingMode.sort, value.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = value.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += value
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { roundingMode }, { value })
}

class KFpRemExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkFpRemDecl(arg0.sort, arg1.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpRoundToIntegralExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val value: KExpr<S>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, value).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpRoundToIntegralDecl(roundingMode.sort, value.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = value.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += value
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { roundingMode }, { value })
}

class KFpMinExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkFpMinDecl(arg0.sort, arg1.sort)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpMaxExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<S, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<S>
        get() = ctx.mkFpMaxDecl(arg0.sort, arg1.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override val sort: S
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): S = arg0.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += arg0
    }

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpLessOrEqualExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpLessOrEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpLessExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpLessDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpGreaterOrEqualExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpGreaterOrEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpGreaterExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpGreaterDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpEqualExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val arg0: KExpr<S>,
    val arg1: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(arg0, arg1)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpEqualDecl(arg0.sort, arg1.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(arg0, arg1)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { arg0 }, { arg1 })
}

class KFpIsNormalExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsNormalDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpIsSubnormalExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsSubnormalDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpIsZeroExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsZeroDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpIsInfiniteExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsInfiniteDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpIsNaNExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsNaNDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpIsNegativeExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsNegativeDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpIsPositiveExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBoolSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBoolSort>
        get() = ctx.mkFpIsPositiveDecl(value.sort)

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpToBvExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val value: KExpr<S>,
    val bvSize: Int,
    val isSigned: Boolean
) : KApp<KBvSort, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, value).uncheckedCast()

    override val decl: KDecl<KBvSort>
        get() = ctx.mkFpToBvDecl(roundingMode.sort, value.sort, bvSize, isSigned)

    override val sort: KBvSort
        get() = ctx.mkBvSort(bvSize.toUInt())

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, value, bvSize, isSigned)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { value }, { bvSize }, { isSigned })
}

class KFpToRealExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KRealSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KRealSort>
        get() = ctx.mkFpToRealDecl(value.sort)

    override val sort: KRealSort
        get() = ctx.realSort

    override fun accept(transformer: KTransformerBase): KExpr<KRealSort> = transformer.transform(this)

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpToIEEEBvExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    val value: KExpr<S>
) : KApp<KBvSort, S>(ctx) {
    override val args: List<KExpr<S>>
        get() = listOf(value)

    override val decl: KDecl<KBvSort>
        get() = ctx.mkFpToIEEEBvDecl(value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<KBvSort> = transformer.transform(this)

    override val sort: KBvSort
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): KBvSort =
        value.sort.let { ctx.mkBvSort(it.significandBits + it.exponentBits) }

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += value
    }

    override fun customHashCode(): Int = hash(value)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { value })
}

class KFpFromBvExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    override val sort: S,
    val sign: KExpr<KBv1Sort>,
    val biasedExponent: KExpr<out KBvSort>,
    val significand: KExpr<out KBvSort>,
) : KApp<S, KBvSort>(ctx) {
    override val args: List<KExpr<KBvSort>>
        get() = listOf(sign, biasedExponent, significand).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpFromBvDecl(sign.sort, biasedExponent.sort, significand.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(sort, sign, biasedExponent, significand)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { sort }, { sign }, { biasedExponent }, { significand })
}

class KFpToFpExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    override val sort: S,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val value: KExpr<out KFpSort>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, value).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkFpToFpDecl(sort, roundingMode.sort, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, value, sort)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { roundingMode }, { value }, { sort })
}

class KRealToFpExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    override val sort: S,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val value: KExpr<KRealSort>
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, value).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkRealToFpDecl(sort, roundingMode.sort, value.sort)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, value, sort)
    override fun customEquals(other: Any): Boolean = structurallyEqual(other, { roundingMode }, { value }, { sort })
}

class KBvToFpExpr<S : KFpSort> internal constructor(
    ctx: KContext,
    override val sort: S,
    val roundingMode: KExpr<KFpRoundingModeSort>,
    val value: KExpr<KBvSort>,
    val signed: Boolean
) : KApp<S, KSort>(ctx) {
    override val args: List<KExpr<KSort>>
        get() = listOf(roundingMode, value).uncheckedCast()

    override val decl: KDecl<S>
        get() = ctx.mkBvToFpDecl(sort, roundingMode.sort, value.sort, signed)

    override fun accept(transformer: KTransformerBase): KExpr<S> = transformer.transform(this)

    override fun customHashCode(): Int = hash(roundingMode, value, signed, sort)
    override fun customEquals(other: Any): Boolean =
        structurallyEqual(other, { roundingMode }, { value }, { signed }, { sort})
}
