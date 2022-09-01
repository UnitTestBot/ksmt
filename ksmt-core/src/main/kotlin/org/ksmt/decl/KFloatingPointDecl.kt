package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp16Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpSort
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
