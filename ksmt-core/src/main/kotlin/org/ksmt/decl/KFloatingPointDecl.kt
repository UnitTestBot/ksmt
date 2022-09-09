package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpRoundingModeSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KRealSort
import org.ksmt.utils.getHalfPrecisionExponent
import org.ksmt.utils.booleanSignBit
import org.ksmt.utils.getExponent
import org.ksmt.utils.halfPrecisionSignificand
import org.ksmt.utils.significand
import org.ksmt.utils.toBinary

abstract class KFpDecl<T : KFpSort, N : Number> internal constructor(
    ctx: KContext,
    sort: T,
    val sign: Boolean,
    val significand: N,
    val exponent: N
) : KConstDecl<T>(
    ctx,
    constructNameForDeclaration(sign, sort, exponent, significand),
    sort
)

private fun <N : Number, T : KFpSort> constructNameForDeclaration(
    sign: Boolean,
    sort: T,
    exponent: N,
    significand: N
): String {
    val exponentBits = sort.exponentBits
    val binaryExponent = exponent.toBinary().takeLast(exponentBits.toInt())
    val significandBits = sort.significandBits
    val binarySignificand = significand
        .toBinary()
        .takeLast(significandBits.toInt() - 1)
        .let { it.padStart(significandBits.toInt() - 1, it[0]) }

    return "FP (sign $sign) ($exponentBits $binaryExponent) ($significandBits $binarySignificand)"
}

class KFp16Decl internal constructor(ctx: KContext, val value: Float) :
    KFpDecl<KFp16Sort, Int>(
        ctx,
        ctx.mkFp16Sort(),
        value.booleanSignBit,
        value.halfPrecisionSignificand,
        value.getHalfPrecisionExponent(isBiased = false)
    ) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp16Sort, *> = ctx.mkFp16(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp32Decl internal constructor(ctx: KContext, val value: Float) :
    KFpDecl<KFp32Sort, Int>(
        ctx,
        ctx.mkFp32Sort(),
        value.booleanSignBit,
        value.significand,
        value.getExponent(isBiased = false)
    ) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp32Sort, *> = ctx.mkFp32(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFp64Decl internal constructor(ctx: KContext, val value: Double) :
    KFpDecl<KFp64Sort, Long>(
        ctx,
        ctx.mkFp64Sort(),
        value.booleanSignBit,
        value.significand,
        value.getExponent(isBiased = false)
    ) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp64Sort, *> = ctx.mkFp64(value)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

// TODO replace significand with bit vector and change KFpDecl accordingly
class KFp128Decl internal constructor(
    ctx: KContext,
    significand: Long,
    exponent: Long,
    signBit: Boolean
) : KFpDecl<KFp128Sort, Long>(ctx, ctx.mkFp128Sort(), signBit, significand, exponent) {
    override fun apply(args: List<KExpr<*>>): KApp<KFp128Sort, *> = ctx.mkFp128(significand, exponent, sign)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KFpCustomSizeDecl internal constructor(
    ctx: KContext,
    significandSize: UInt,
    exponentSize: UInt,
    significand: Long,
    exponent: Long,
    signBit: Boolean
) : KFpDecl<KFpSort, Long>(ctx, ctx.mkFpSort(exponentSize, significandSize), signBit, significand, exponent) {
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

class KFpAbsDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<T, T>(ctx, "fp.abs", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, KExpr<T>> = ctx.mkFpAbsExpr(arg)
}

class KFpNegationDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<T, T>(ctx, "fp.neg", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, KExpr<T>> = ctx.mkFpNegationExpr(arg)
}

class KFpAddDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, R, T, T>(ctx, "fp.add", arg0Sort, roundingModeSort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<R>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpAddExpr(arg0, arg1, arg2)
}

class KFpSubDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, R, T, T>(ctx, "fp.sub", arg0Sort, roundingModeSort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<R>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpSubExpr(arg0, arg1, arg2)
}

class KFpMulDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, R, T, T>(ctx, "fp.mul", arg0Sort, roundingModeSort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<R>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpMulExpr(arg0, arg1, arg2)
}

class KFpDivDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl3<T, R, T, T>(ctx, "fp.div", arg0Sort, roundingModeSort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<R>,
        arg1: KExpr<T>,
        arg2: KExpr<T>
    ): KApp<T, *> = ctx.mkFpDivExpr(arg0, arg1, arg2)
}

class KFpFusedMulAddDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext, roundingModeSort: R, arg0Sort: T, arg1Sort: T, arg2Sort: T
) : KFuncDecl4<T, R, T, T, T>(ctx, "fp.fma", arg0Sort, roundingModeSort, arg0Sort, arg1Sort, arg2Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<R>,
        arg1: KExpr<T>,
        arg2: KExpr<T>,
        arg3: KExpr<T>
    ): KApp<T, *> = ctx.mkFpFusedMulAddExpr(arg0, arg1, arg2, arg3)
}

class KFpSqrtDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    valueSort: T
) : KFuncDecl2<T, R, T>(ctx, "fp.sqrt", resultSort = valueSort, roundingModeSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<R>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpSqrtExpr(arg0, arg1)

}

class KFpRemDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<T, T, T>(ctx, "fp.rem", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpRemExpr(arg0, arg1)
}

class KFpRoundToIntegralDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    valueSort: T
) : KFuncDecl2<T, R, T>(ctx, "fp.roundToIntegral", resultSort = valueSort, roundingModeSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<R>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpRoundToIntegralExpr(arg0, arg1)
}

class KFpMinDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<T, T, T>(ctx, "fp.min", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpMinExpr(arg0, arg1)
}

class KFpMaxDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<T, T, T>(ctx, "fp.max", arg0Sort, arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = ctx.mkFpMaxExpr(arg0, arg1)
}

class KFpLessOrEqualDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "fp.leq", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> =
        ctx.mkFpLessOrEqualExpr(arg0, arg1)
}

class KFpLessDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "fp.lt", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = ctx.mkFpLessExpr(arg0, arg1)
}

class KFpGreaterOrEqualDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "fp.geq", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> =
        ctx.mkFpGreaterOrEqualExpr(arg0, arg1)
}

class KFpGreaterDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "fp.gt", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = ctx.mkFpGreaterExpr(arg0, arg1)
}

class KFpEqualDecl<T : KFpSort> internal constructor(ctx: KContext, arg0Sort: T, arg1Sort: T) :
    KFuncDecl2<KBoolSort, T, T>(ctx, "fp.eq", ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<KBoolSort, *> = ctx.mkFpEqualExpr(arg0, arg1)
}

class KFpIsNormalDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isNormal", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsNormalExpr(arg)
}

class KFpIsSubnormalDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isSubnormal", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsSubnormalExpr(arg)
}

class KFpIsZeroDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isZero", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsZeroExpr(arg)
}

class KFpIsInfiniteDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isInfinite", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsInfiniteExpr(arg)
}

class KFpIsNaNDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isNaN", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsNaNExpr(arg)
}

class KFpIsNegativeDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isNegative", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsNegativeExpr(arg)
}

class KFpIsPositiveDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBoolSort, T>(ctx, "fp.isPositive", ctx.mkBoolSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBoolSort, KExpr<T>> = ctx.mkFpIsPositiveExpr(arg)
}

class KFpToBvDecl<R : KFpRoundingModeSort, T : KFpSort> internal constructor(
    ctx: KContext,
    roundingModeSort: R,
    valueSort: T,
    val bvSize: Int,
    val isSigned: Boolean
) : KFuncDecl2<KBvSort, R, T>(
    ctx,
    "fp.to_${if (isSigned) "s" else "u"}bv",
    ctx.mkBvSort(bvSize.toUInt()),
    roundingModeSort,
    valueSort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<R>, arg1: KExpr<T>): KApp<KBvSort, *> =
        ctx.mkFpToBvExpr(arg0, arg1, bvSize, isSigned)
}

class KFpToRealDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KRealSort, T>(ctx, "fp.to_real", ctx.mkRealSort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KRealSort, KExpr<T>> = ctx.mkFpToRealExpr(arg)
}

class KFpToIEEEBvDecl<T : KFpSort> internal constructor(ctx: KContext, valueSort: T) :
    KFuncDecl1<KBvSort, T>(
        ctx,
        "fp.to_ieee_bv",
        ctx.mkBvSort(valueSort.significandBits + valueSort.exponentBits),
        valueSort
    ) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBvSort, KExpr<T>> = ctx.mkFpToIEEEBvExpr(arg)
}
