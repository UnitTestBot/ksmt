package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KBitVecValue
import org.ksmt.expr.KExpr
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
import org.ksmt.utils.getHalfPrecisionExponent
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.getExponent
import org.ksmt.utils.halfPrecisionSignificand
import org.ksmt.utils.significand
import org.ksmt.utils.toBinary

abstract class KFpDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    sort: T,
    val sign: Boolean,
    val significandBinary: String,
    val exponentBinary: String
) : KConstDecl<T>(
    ctx,
    constructNameForDeclaration(sign, sort, exponentBinary, significandBinary),
    sort
)

private fun <T : KFpSort> constructNameForDeclaration(
    sign: Boolean,
    sort: T,
    exponent: String,
    significand: String
): String {
    val exponentBits = sort.exponentBits
    val binaryExponent = exponent.takeLast(exponentBits.toInt())
    val significandBits = sort.significandBits
    val binarySignificand = significand
        .takeLast(significandBits.toInt() - 1)
        .let { it.padStart(significandBits.toInt() - 1, it[0]) }

    return "FP (sign $sign) ($exponentBits $binaryExponent) ($significandBits $binarySignificand)"
}

class KFp16Decl internal constructor(
    ctx: KContext,
    val value: Float
) : KFpDecl<KFp16Sort>(
    ctx,
    ctx.mkFp16Sort(),
    value.booleanSignBit,
    value.halfPrecisionSignificand.toBinary(),
    value.getHalfPrecisionExponent(isBiased = false).toBinary()
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp16Sort, *> = ctx.mkFp16(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp32Decl internal constructor(
    ctx: KContext,
    val value: Float
) : KFpDecl<KFp32Sort>(
    ctx,
    ctx.mkFp32Sort(),
    value.booleanSignBit,
    value.significand.toBinary(),
    value.getExponent(isBiased = false).toBinary()
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp32Sort, *> = ctx.mkFp32(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp64Decl internal constructor(
    ctx: KContext,
    val value: Double
) : KFpDecl<KFp64Sort>(
    ctx,
    ctx.mkFp64Sort(),
    value.booleanSignBit,
    value.significand.toBinary(),
    value.getExponent(isBiased = false).toBinary()
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp64Sort, *> = ctx.mkFp64(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp128Decl internal constructor(
    ctx: KContext,
    val significand: KBitVecValue<*>,
    val exponent: KBitVecValue<*>,
    signBit: Boolean
) : KFpDecl<KFp128Sort>(ctx,
    ctx.mkFp128Sort(),
    signBit,
    significand.stringValue,
    exponent.stringValue
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp128Sort, *> = ctx.mkFp128(significand, exponent, sign)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpCustomSizeDecl internal constructor(
    ctx: KContext,
    significandSize: UInt,
    exponentSize: UInt,
    val significand: KBitVecValue<*>,
    val exponent: KBitVecValue<*>,
    signBit: Boolean
) : KFpDecl<KFpSort>(
    ctx,
    ctx.mkFpSort(exponentSize, significandSize),
    signBit,
    significand.stringValue,
    exponent.stringValue
) {
    override fun apply(args: List<KExpr<*>>): KApp<KFpSort, *> =
        ctx.mkFpCustomSize(
            sort.exponentBits,
            sort.significandBits,
            exponent,
            significand,
            sign
        )

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpAbsDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "fp.abs", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, KExpr<T>> = ctx.mkFpAbsExpr(arg)
}

class KFpNegationDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "fp.neg", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, KExpr<T>> = ctx.mkFpNegationExpr(arg)
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
    ): KApp<T, *> = ctx.mkFpAddExpr(arg0, arg1, arg2)
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
    ): KApp<T, *> = ctx.mkFpSubExpr(arg0, arg1, arg2)
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
        ctx.mkFpMulExpr(arg0, arg1, arg2)
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
    ): KApp<T, *> = ctx.mkFpDivExpr(arg0, arg1, arg2)
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
    ): KApp<T, *> = ctx.mkFpFusedMulAddExpr(arg0, arg1, arg2, arg3)
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
    ): KApp<T, *> = ctx.mkFpSqrtExpr(arg0, arg1)
}

class KFpRemDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "fp.rem", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpRemExpr(arg0, arg1)
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
    ): KApp<T, *> = ctx.mkFpRoundToIntegralExpr(arg0, arg1)
}

class KFpMinDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "fp.min", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpMinExpr(arg0, arg1)
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
    ): KApp<T, *> = ctx.mkFpMaxExpr(arg0, arg1)
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
    ): KApp<KBoolSort, *> = ctx.mkFpLessOrEqualExpr(arg0, arg1)
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
    ): KApp<KBoolSort, *> = ctx.mkFpLessExpr(arg0, arg1)
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
    ): KApp<KBoolSort, *> = ctx.mkFpGreaterOrEqualExpr(arg0, arg1)
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
    ): KApp<KBoolSort, *> = ctx.mkFpGreaterExpr(arg0, arg1)
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
    ): KApp<KBoolSort, *> = ctx.mkFpEqualExpr(arg0, arg1)
}

class KFpIsNormalDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isNormal", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsNormalExpr(arg)
}

class KFpIsSubnormalDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isSubnormal", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsSubnormalExpr(arg)
}

class KFpIsZeroDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isZero", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsZeroExpr(arg)
}

class KFpIsInfiniteDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isInfinite", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsInfiniteExpr(arg)
}

class KFpIsNaNDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isNaN", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsNaNExpr(arg)
}

class KFpIsNegativeDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isNegative", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsNegativeExpr(arg)
}

class KFpIsPositiveDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBoolSort, T>(ctx, "fp.isPositive", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsPositiveExpr(arg)
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
    ): KApp<KBvSort, *> = ctx.mkFpToBvExpr(arg0, arg1, bvSize, isSigned)
}

class KFpToRealDecl<T : KFpSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KRealSort, T>(ctx, "fp.to_real", ctx.mkRealSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KRealSort, KExpr<T>> = ctx.mkFpToRealExpr(arg)
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

    override fun KContext.apply(arg: KExpr<T>): KApp<KBvSort, KExpr<T>> = ctx.mkFpToIEEEBvExpr(arg)
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
    ): KApp<T, *> = ctx.mkFpFromBvExpr(arg0, arg1, arg2)
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
    ): KApp<T, *> = ctx.mkFpToFpExpr(sort, arg0, arg1)
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
    ): KApp<T, *> = ctx.mkRealToFpExpr(sort, arg0, arg1)
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
    ): KApp<T, *> = ctx.mkBvToFpExpr(sort, arg0, arg1, isSigned)
}
