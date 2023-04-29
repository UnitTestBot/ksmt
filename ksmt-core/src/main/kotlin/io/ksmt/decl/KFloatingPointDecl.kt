package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KBitVecValue
import io.ksmt.expr.KExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpRoundingModeSort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KRealSort
import io.ksmt.sort.KSort
import io.ksmt.utils.getHalfPrecisionExponent
import io.ksmt.utils.booleanSignBit
import io.ksmt.utils.getExponent
import io.ksmt.utils.halfPrecisionSignificand
import io.ksmt.utils.significand
import io.ksmt.utils.toBinary

abstract class KFpDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    sort: T,
    val sign: KDecl<KBv1Sort>,
    val significandBinary: KDecl<out KBvSort>,
    val unbiasedExponentBinary: KDecl<out KBvSort>
) : KConstDecl<T>(
    ctx,
    "(fp ${sign.name} ${unbiasedExponentBinary.name} ${significandBinary.name})",
    sort
) {
    companion object {
        internal fun KContext.fpExponentDecl(exponentBits: UInt, exponentBinary: String) =
            mkBvDecl(exponentBinary.takeLast(exponentBits.toInt()), exponentBits)

        internal fun KContext.fpSignificandDecl(significandBits: UInt, significandBinary: String): KDecl<KBvSort> {
            val normalizedSignificandBits = significandBits - 1u
            val normalizedSignificandBinary = significandBinary
                .takeLast(normalizedSignificandBits.toInt())
                .let { it.padStart(normalizedSignificandBits.toInt(), it[0]) }
            return mkBvDecl(normalizedSignificandBinary, normalizedSignificandBits)
        }
    }
}

class KFp16Decl internal constructor(
    ctx: KContext,
    val value: Float
) : KFpDecl<KFp16Sort>(
    ctx = ctx,
    sort = ctx.mkFp16Sort(),
    sign = ctx.mkBvDecl(value.booleanSignBit),
    significandBinary = ctx.fpSignificandDecl(
        KFp16Sort.significandBits,
        value.halfPrecisionSignificand.toBinary()
    ),
    unbiasedExponentBinary = ctx.fpExponentDecl(
        KFp16Sort.exponentBits,
        value.getHalfPrecisionExponent(isBiased = false).toBinary()
    )
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp16Sort, *> = ctx.mkFp16(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp32Decl internal constructor(
    ctx: KContext,
    val value: Float
) : KFpDecl<KFp32Sort>(
    ctx = ctx,
    sort = ctx.mkFp32Sort(),
    sign = ctx.mkBvDecl(value.booleanSignBit),
    significandBinary = ctx.fpSignificandDecl(
        KFp32Sort.significandBits,
        value.significand.toBinary()
    ),
    unbiasedExponentBinary = ctx.fpExponentDecl(
        KFp32Sort.exponentBits,
        value.getExponent(isBiased = false).toBinary()
    )
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp32Sort, *> = ctx.mkFp32(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp64Decl internal constructor(
    ctx: KContext,
    val value: Double
) : KFpDecl<KFp64Sort>(
    ctx = ctx,
    sort = ctx.mkFp64Sort(),
    sign = ctx.mkBvDecl(value.booleanSignBit),
    significandBinary = ctx.fpSignificandDecl(
        KFp64Sort.significandBits,
        value.significand.toBinary()
    ),
    unbiasedExponentBinary = ctx.fpExponentDecl(
        KFp64Sort.exponentBits,
        value.getExponent(isBiased = false).toBinary()
    )
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp64Sort, *> = ctx.mkFp64(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp128Decl internal constructor(
    ctx: KContext,
    val significand: KBitVecValue<*>,
    val unbiasedExponent: KBitVecValue<*>,
    val signBit: Boolean
) : KFpDecl<KFp128Sort>(
    ctx = ctx,
    sort = ctx.mkFp128Sort(),
    sign = ctx.mkBvDecl(signBit),
    significandBinary = significand.decl,
    unbiasedExponentBinary = unbiasedExponent.decl
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp128Sort, *> = ctx.mkFp128(significand, unbiasedExponent, signBit)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpCustomSizeDecl internal constructor(
    ctx: KContext,
    significandSize: UInt,
    exponentSize: UInt,
    val significand: KBitVecValue<*>,
    val unbiasedExponent: KBitVecValue<*>,
    val signBit: Boolean
) : KFpDecl<KFpSort>(
    ctx = ctx,
    sort = ctx.mkFpSort(exponentSize, significandSize),
    sign = ctx.mkBvDecl(signBit),
    significandBinary = significand.decl,
    unbiasedExponentBinary = unbiasedExponent.decl
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpSort, *> =
        ctx.mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            unbiasedExponent,
            significand,
            signBit
        )

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpAbsDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "fp.abs", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, T> = ctx.mkFpAbsExprNoSimplify(arg)
}

class KFpNegationDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "fp.neg", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, T> = ctx.mkFpNegationExprNoSimplify(arg)
}

class KFpAddDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, KFpRoundingModeSort, T, T>(
    ctx,
    "fp.add",
    arg0Sort,
    roundingModeSort,
    arg0Sort,
    arg1Sort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpAddExprNoSimplify(arg0, arg1, arg2)
}

class KFpSubDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, KFpRoundingModeSort, T, T>(
    ctx,
    "fp.sub",
    resultSort = arg0Sort,
    roundingModeSort,
    arg0Sort,
    arg1Sort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpSubExprNoSimplify(arg0, arg1, arg2)
}

class KFpMulDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, KFpRoundingModeSort, T, T>(
    ctx, "fp.mul", resultSort = arg0Sort, roundingModeSort, arg0Sort, arg1Sort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KFpRoundingModeSort>, arg1: KExpr<T>, arg2: KExpr<T>): KApp<T, *> =
        ctx.mkFpMulExprNoSimplify(arg0, arg1, arg2)
}

class KFpDivDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, KFpRoundingModeSort, T, T>(
    ctx,
    "fp.div",
    resultSort = arg0Sort,
    roundingModeSort,
    arg0Sort,
    arg1Sort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpDivExprNoSimplify(arg0, arg1, arg2)
}

class KFpFusedMulAddDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    arg0Sort: T,
    arg1Sort: T,
    arg2Sort: T
) : KFuncDecl4<T, KFpRoundingModeSort, T, T, T>(
    ctx,
    "fp.fma",
    resultSort = arg0Sort,
    roundingModeSort,
    arg0Sort,
    arg1Sort,
    arg2Sort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>,
        arg2: KExpr<T>,
        arg3: KExpr<T>
    ): KApp<T, *> = ctx.mkFpFusedMulAddExprNoSimplify(arg0, arg1, arg2, arg3)
}

class KFpSqrtDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: T
) : KFuncDecl2<T, KFpRoundingModeSort, T>(
    ctx,
    "fp.sqrt",
    resultSort = valueSort,
    roundingModeSort,
    valueSort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>
    ): KApp<T, *> = ctx.mkFpSqrtExprNoSimplify(arg0, arg1)
}

class KFpRemDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "fp.rem", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpRemExprNoSimplify(arg0, arg1)
}

class KFpRoundToIntegralDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: T
) : KFuncDecl2<T, KFpRoundingModeSort, T>(
    ctx,
    "fp.roundToIntegral",
    resultSort = valueSort,
    roundingModeSort,
    valueSort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>
    ): KApp<T, *> = ctx.mkFpRoundToIntegralExprNoSimplify(arg0, arg1)
}

class KFpMinDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "fp.min", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpMinExprNoSimplify(arg0, arg1)
}

class KFpMaxDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "fp.max", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<T, *> = ctx.mkFpMaxExprNoSimplify(arg0, arg1)
}

class KFpLessOrEqualDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "fp.leq", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = ctx.mkFpLessOrEqualExprNoSimplify(arg0, arg1)
}

class KFpLessDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "fp.lt", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = ctx.mkFpLessExprNoSimplify(arg0, arg1)
}

class KFpGreaterOrEqualDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "fp.geq", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = ctx.mkFpGreaterOrEqualExprNoSimplify(arg0, arg1)
}

class KFpGreaterDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "fp.gt", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = ctx.mkFpGreaterExprNoSimplify(arg0, arg1)
}

class KFpEqualDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "fp.eq", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = ctx.mkFpEqualExprNoSimplify(arg0, arg1)
}

class KFpIsNormalDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isNormal", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsNormalExprNoSimplify(arg)
}

class KFpIsSubnormalDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isSubnormal", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsSubnormalExprNoSimplify(arg)
}

class KFpIsZeroDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isZero", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsZeroExprNoSimplify(arg)
}

class KFpIsInfiniteDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isInfinite", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsInfiniteExprNoSimplify(arg)
}

class KFpIsNaNDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isNaN", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsNaNExprNoSimplify(arg)
}

class KFpIsNegativeDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isNegative", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsNegativeExprNoSimplify(arg)
}

class KFpIsPositiveDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isPositive", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, T> = ctx.mkFpIsPositiveExprNoSimplify(arg)
}

class KFpToBvDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: T,
    val bvSize: Int,
    val isSigned: Boolean
) : KFuncDecl2<KBvSort, KFpRoundingModeSort, T>(
    ctx,
    "fp.to_${if (isSigned) "s" else "u"}bv",
    ctx.mkBvSort(bvSize.toUInt()),
    roundingModeSort,
    valueSort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<T>
    ): KApp<KBvSort, *> = ctx.mkFpToBvExprNoSimplify(arg0, arg1, bvSize, isSigned)
}

class KFpToRealDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KRealSort, T>(ctx, "fp.to_real", ctx.mkRealSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KRealSort, T> = ctx.mkFpToRealExprNoSimplify(arg)
}

class KFpToIEEEBvDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBvSort, T>(
    ctx,
    "fp.to_ieee_bv",
    ctx.mkBvSort(valueSort.significandBits + valueSort.exponentBits),
    valueSort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBvSort, T> = ctx.mkFpToIEEEBvExprNoSimplify(arg)
}

class KFpFromBvDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    sort: T,
    signSort: KBv1Sort,
    expSort: KBvSort,
    significandSort: KBvSort
) : KFuncDecl3<T, KBv1Sort, KBvSort, KBvSort>(
    ctx,
    "fp.to_fp",
    sort,
    signSort,
    expSort,
    significandSort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBv1Sort>,
        arg1: KExpr<KBvSort>,
        arg2: KExpr<KBvSort>
    ): KApp<T, *> = ctx.mkFpFromBvExprNoSimplify(arg0, arg1, arg2)
}

abstract class KToFpDecl<T : KFpSort, S : KSort> internal constructor(
    ctx: KContext,
    sort: T,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: S
) : KFuncDecl2<T, KFpRoundingModeSort, S>(ctx, "fp.to_fp", resultSort = sort, roundingModeSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpToFpDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    sort: T,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: KFpSort
) : KToFpDecl<T, KFpSort>(ctx, sort, roundingModeSort, valueSort) {
    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<KFpSort>
    ): KApp<T, *> = ctx.mkFpToFpExprNoSimplify(sort, arg0, arg1)
}

class KRealToFpDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    sort: T,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: KRealSort
) : KToFpDecl<T, KRealSort>(ctx, sort, roundingModeSort, valueSort) {
    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<KRealSort>
    ): KApp<T, *> = ctx.mkRealToFpExprNoSimplify(sort, arg0, arg1)
}

class KBvToFpDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    sort: T,
    roundingModeSort: KFpRoundingModeSort,
    valueSort: KBvSort,
    val isSigned: Boolean
) : KToFpDecl<T, KBvSort>(ctx, sort, roundingModeSort, valueSort) {
    override fun KContext.apply(
        arg0: KExpr<KFpRoundingModeSort>,
        arg1: KExpr<KBvSort>
    ): KApp<T, *> = ctx.mkBvToFpExprNoSimplify(sort, arg0, arg1, isSigned)
}
