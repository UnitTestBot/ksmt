package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.expr.rewrite.simplify.simplifyBvExtractExpr
import org.ksmt.expr.rewrite.simplify.simplifyFpFromBvExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.symfpu.UnpackedFp.Companion.iteOp
import org.ksmt.symfpu.UnpackedFp.Companion.makeInf
import org.ksmt.symfpu.UnpackedFp.Companion.makeNaN
import org.ksmt.symfpu.UnpackedFp.Companion.makeZero
import org.ksmt.utils.BvUtils.bvZero
import org.ksmt.utils.cast

fun <Fp : KFpSort> KContext.unpack(sort: Fp, packedFloat: KExpr<KBvSort>): UnpackedFp<Fp> {
    val pWidth = packedFloat.sort.sizeBits.toInt()
    val exWidth = sort.exponentBits.toInt()

    // Extract
    val packedSignificand = mkBvExtractExpr(pWidth - exWidth - 2, 0, packedFloat)
    val packedExponent = mkBvExtractExpr(pWidth - 2, pWidth - exWidth - 1, packedFloat)
    val sign = bvToBool(mkBvExtractExpr(pWidth - 1, pWidth - 1, packedFloat))
    return unpack(sort, sign, packedExponent, packedSignificand)
}

fun <Fp : KFpSort> KContext.unpack(
    sort: Fp,
    sign: KExpr<KBoolSort>,
    packedExponent: KExpr<KBvSort>,
    packedSignificand: KExpr<KBvSort>,
//    packedFloat: KExpr<KBvSort> = mkBvConcatExpr(boolToBv(sign), packedExponent, packedSignificand)
): UnpackedFp<Fp> {
    val unpackedExWidth = exponentWidth(sort)

    val exponent = mkBvSubExpr(
        mkBvZeroExtensionExpr(unpackedExWidth - sort.exponentBits.toInt(), packedExponent),
        mkBv(sort.exponentShiftSize(), unpackedExWidth.toUInt())
    )

    val significandWithLeadingZero = mkBvConcatExpr(bvZero(), packedSignificand)
    val significandWithLeadingOne = mkBvConcatExpr(bvOne(), packedSignificand)

    val packedFp = UnpackedFp.PackedFp.Exists(sign, packedExponent, packedSignificand)
    val ufNormal = UnpackedFp(this, sort, sign, exponent, significandWithLeadingOne, packedFp)
    val ufSubnormalBase = UnpackedFp(this, sort, sign, minNormalExponent(sort), significandWithLeadingZero, packedFp)

    // Analyse
    val zeroExponent = isAllZeros(packedExponent)
    val onesExponent = isAllOnes(packedExponent)
    val zeroSignificand = isAllZeros(packedSignificand)

    // Identify the cases
    val isZero = zeroExponent and zeroSignificand
    val isSubnormal = zeroExponent and !zeroSignificand
    val isInf = onesExponent and zeroSignificand
    val isNaN = onesExponent and !zeroSignificand


    return iteOp(
        isNaN, makeNaN(sort), iteOp(
            isInf, makeInf(sort, sign), iteOp(
                isZero, makeZero(sort, sign), iteOp(
                    !isSubnormal, ufNormal, ufSubnormalBase.normaliseUp()
                )
            )
        )
    )
}

fun <Fp : KFpSort> KContext.packToBv(uf: UnpackedFp<Fp>): KExpr<KBvSort> {
    uf.packedBv.toIEEE()?.let { return it }


    // Sign
    val packedSign = uf.signBv()
    // Exponent
    val packedExWidth = uf.sort.exponentBits.toInt()
    val inNormalRange = uf.inNormalRange()
    val inSubnormalRange = !inNormalRange
    val biasedExp = mkBvAddExpr(uf.unbiasedExponent, bias(uf.sort))
    // Will be correct for normal values only, subnormals may still be negative.
    val packedBiasedExp = mkBvExtractExpr(packedExWidth - 1, 0, biasedExp)
    val maxExp = ones(packedExWidth.toUInt())
    val minExp = bvZero(packedExWidth.toUInt())

    val hasMaxExp = uf.isNaN or uf.isInf
    val hasMinExp = uf.isZero or inSubnormalRange
    val hasFixedExp = hasMaxExp or hasMinExp

    val packedExp = mkIte(hasFixedExp, mkIte(hasMaxExp, maxExp, minExp.cast()), packedBiasedExp)

    // Significand
    val packedSigWidth = uf.sort.significandBits.toInt() - 1
    val unpackedSignificand = uf.normalizedSignificand
    val unpackedSignificandWidth = uf.normalizedSignificand.sort.sizeBits.toInt()

    check(packedSigWidth == unpackedSignificandWidth - 1)
    val dropLeadingOne = mkBvExtractExpr(packedSigWidth - 1, 0, unpackedSignificand)

    // The amount needed to normalise the number
    val subnormalShiftAmount = max(
        mkBvSubExpr(minNormalExponent(uf.sort), uf.unbiasedExponent), // minNormalExponent - exponent
        bvZero(uf.unbiasedExponent.sort.sizeBits).cast()
    )


    val shiftAmount = if (subnormalShiftAmount.sort.sizeBits.toInt() <= unpackedSignificandWidth) {
        subnormalShiftAmount.matchWidthUnsigned(unpackedSignificand)
    } else {
        mkBvExtractExpr(unpackedSignificandWidth - 1, 0, subnormalShiftAmount)
    }
    // The extraction could lose data if exponent is much larger than the significand
    // However getSubnormalAmount is between 0 and packedSigWidth, so it should be safe
    val correctedSubnormal =
        mkBvExtractExpr(packedSigWidth - 1, 0, mkBvLogicalShiftRightExpr(unpackedSignificand, shiftAmount))
    val hasFixedSignificand = uf.isNaN or uf.isInf or uf.isZero

    val nanBv = leadingOne(packedSigWidth)
    val packedSig = mkIte(
        hasFixedSignificand, mkIte(
            uf.isNaN, nanBv, bvZero(packedSigWidth.toUInt()).cast()
        ), mkIte(
            inNormalRange, dropLeadingOne, correctedSubnormal
        )
    )

    return mkBvConcatExpr(packedSign, packedExp, packedSig)
}


fun <Fp : KFpSort> KContext.pack(packed: KExpr<KBvSort>, sort: Fp): KExpr<Fp> {
    check(packed.sort.sizeBits == sort.exponentBits + sort.significandBits)

    val pWidth = packed.sort.sizeBits.toInt()
    val exWidth = sort.exponentBits.toInt()

    // Extract
    val packedSignificand = simplifyBvExtractExpr(pWidth - exWidth - 2, 0, packed)
    val packedExponent = simplifyBvExtractExpr(pWidth - 2, pWidth - exWidth - 1, packed)
    val sign = simplifyBvExtractExpr(pWidth - 1, pWidth - 1, packed)

    return simplifyFpFromBvExpr(sign.cast(), packedExponent, packedSignificand)
}

fun KExpr<KBvSort>.matchWidthUnsigned(expr: KExpr<KBvSort>): KExpr<KBvSort> {
    check(sort.sizeBits.toInt() <= expr.sort.sizeBits.toInt())
    if (sort.sizeBits.toInt() == expr.sort.sizeBits.toInt()) return this
    return expr.ctx.mkBvZeroExtensionExpr(expr.sort.sizeBits.toInt() - sort.sizeBits.toInt(), this)
}

fun KExpr<KBvSort>.matchWidthSigned(ctx: KContext, expr: KExpr<KBvSort>): KExpr<KBvSort> {
    check(sort.sizeBits.toInt() <= expr.sort.sizeBits.toInt())
    if (sort.sizeBits.toInt() == expr.sort.sizeBits.toInt()) return this
    return ctx.mkBvSignExtensionExpr(expr.sort.sizeBits.toInt() - sort.sizeBits.toInt(), this)
}
