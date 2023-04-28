package io.ksmt.decl

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.expr.printer.BvValuePrintMode
import io.ksmt.sort.KBv16Sort
import io.ksmt.sort.KBv32Sort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.sort.KBvSort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KBv1Sort
import io.ksmt.sort.KIntSort
import java.math.BigInteger

abstract class KBitVecValueDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    val value: Number,
    sort: T
) : KConstDecl<T>(ctx, mkDeclName(ctx, value, sort.sizeBits), sort) {
    companion object {
        private fun mkDeclName(ctx: KContext, value: Number, sizeBits: UInt): String =
            if (ctx.printerParams.bvValuePrintMode == BvValuePrintMode.HEX && sizeBits % HALF_BYTE == 0u) {
                /**
                 * According to the smt-lib standard, we can use hex notation
                 * `#x` only if the number of bits in Bv is aligned with the number of bits
                 * represented by a single character in hexadecimal format ([HALF_BYTE]).
                 * */
                "#x${value.toHexString(sizeBits / HALF_BYTE)}"
            } else {
                "#b${value.toBinaryString(sizeBits)}"
            }

        private const val HEX_RADIX = 16
        private const val BINARY_RADIX = 2
        private const val HALF_BYTE = 4u

        private fun Number.toHexString(sizeHalfBytes: UInt) = when (this) {
            is Byte -> toUByte().toString(HEX_RADIX)
            is Short -> toUShort().toString(HEX_RADIX)
            is Int -> toUInt().toString(HEX_RADIX)
            is Long -> toULong().toString(HEX_RADIX)
            is BigInteger -> toString(HEX_RADIX)
            else -> error("Unsupported type for transformation into a hex string: ${this::class.simpleName}")
        }.padStart(sizeHalfBytes.toInt(), '0')

        private fun Number.toBinaryString(sizeBits: UInt) = when (this) {
            is Byte -> toUByte().toString(BINARY_RADIX)
            is Short -> toUShort().toString(BINARY_RADIX)
            is Int -> toUInt().toString(BINARY_RADIX)
            is Long -> toULong().toString(BINARY_RADIX)
            is BigInteger -> toString(BINARY_RADIX)
            else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
        }.padStart(sizeBits.toInt(), '0')
    }
}

class KBitVec1ValueDecl internal constructor(
    ctx: KContext,
    val boolValue: Boolean
) : KBitVecValueDecl<KBv1Sort>(
    ctx,
    value = if (boolValue) 1 else 0,
    ctx.mkBv1Sort()
) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv1Sort, *> = ctx.mkBv(boolValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec8ValueDecl internal constructor(
    ctx: KContext,
    val byteValue: Byte
) : KBitVecValueDecl<KBv8Sort>(ctx, byteValue, ctx.mkBv8Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv8Sort, *> = ctx.mkBv(byteValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec16ValueDecl internal constructor(
    ctx: KContext,
    val shortValue: Short
) : KBitVecValueDecl<KBv16Sort>(ctx, shortValue, ctx.mkBv16Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv16Sort, *> = ctx.mkBv(shortValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec32ValueDecl internal constructor(
    ctx: KContext,
    val intValue: Int
) : KBitVecValueDecl<KBv32Sort>(ctx, intValue, ctx.mkBv32Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv32Sort, *> = ctx.mkBv(intValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec64ValueDecl internal constructor(
    ctx: KContext,
    val longValue: Long
) : KBitVecValueDecl<KBv64Sort>(ctx, longValue, ctx.mkBv64Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv64Sort, *> = ctx.mkBv(longValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVecCustomSizeValueDecl internal constructor(
    ctx: KContext,
    val bigIntValue: BigInteger,
    sizeBits: UInt
) : KBitVecValueDecl<KBvSort>(
    ctx,
    value = bigIntValue,
    ctx.mkBvSort(sizeBits)
) {
    override fun apply(args: List<KExpr<*>>): KApp<KBvSort, *> = ctx.mkBv(bigIntValue, sort.sizeBits)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}


// Expressions with bit-vectors
class KBvNotDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "bvnot", resultSort = valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, T> = mkBvNotExprNoSimplify(arg)
}

class KBvReductionAndDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBv1Sort, T>(ctx, "bvredand", resultSort = ctx.mkBv1Sort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBv1Sort, T> = mkBvReductionAndExprNoSimplify(arg)
}

class KBvReductionOrDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<KBv1Sort, T>(ctx, "bvredor", resultSort = ctx.mkBv1Sort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<KBv1Sort, T> = mkBvReductionOrExprNoSimplify(arg)
}

class KBvAndDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvand", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvAndExprNoSimplify(arg0, arg1)
}

class KBvOrDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvOrExprNoSimplify(arg0, arg1)
}

class KBvXorDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvxor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvXorExprNoSimplify(arg0, arg1)
}

class KBvNAndDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvnand", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvNAndExprNoSimplify(arg0, arg1)
}

class KBvNorDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvnor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvNorExprNoSimplify(arg0, arg1)
}

class KBvXNorDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvxnor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvXNorExprNoSimplify(arg0, arg1)
}

class KBvNegationDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "bvneg", resultSort = valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<T>): KApp<T, T> = mkBvNegationExprNoSimplify(arg)
}

class KBvAddDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvadd", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvAddExprNoSimplify(arg0, arg1)
}

class KBvSubDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvsub", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvSubExprNoSimplify(arg0, arg1)
}

class KBvMulDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvmul", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvMulExprNoSimplify(arg0, arg1)
}

class KBvUnsignedDivDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvudiv", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvUnsignedDivExprNoSimplify(arg0, arg1)
}

class KBvSignedDivDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvsdiv", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvSignedDivExprNoSimplify(arg0, arg1)
}

class KBvUnsignedRemDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvurem", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvUnsignedRemExprNoSimplify(arg0, arg1)
}

class KBvSignedRemDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvsrem", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvSignedRemExprNoSimplify(arg0, arg1)
}

class KBvSignedModDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvsmod", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvSignedModExprNoSimplify(arg0, arg1)
}

class KBvUnsignedLessDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvult", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvUnsignedLessExprNoSimplify(arg0, arg1)
}

class KBvSignedLessDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvslt", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvSignedLessExprNoSimplify(arg0, arg1)
}

class KBvSignedLessOrEqualDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvsle", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvSignedLessOrEqualExprNoSimplify(arg0, arg1)
}


class KBvUnsignedLessOrEqualDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvule", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvUnsignedLessOrEqualExprNoSimplify(arg0, arg1)
}

class KBvUnsignedGreaterOrEqualDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvuge", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvUnsignedGreaterOrEqualExprNoSimplify(arg0, arg1)
}

class KBvSignedGreaterOrEqualDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvsge", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvSignedGreaterOrEqualExprNoSimplify(arg0, arg1)
}

class KBvUnsignedGreaterDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvugt", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvUnsignedGreaterExprNoSimplify(arg0, arg1)
}

class KBvSignedGreaterDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(ctx, "bvsgt", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<KBoolSort, *> = mkBvSignedGreaterExprNoSimplify(arg0, arg1)
}

class KBvConcatDecl internal constructor(
    ctx: KContext,
    arg0Sort: KBvSort,
    arg1Sort: KBvSort
) : KFuncDecl2<KBvSort, KBvSort, KBvSort>(
    ctx,
    name = "concat",
    resultSort = ctx.mkBvSort(arg0Sort.sizeBits + arg1Sort.sizeBits),
    arg0Sort,
    arg1Sort
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvConcatExprNoSimplify(arg0, arg1)
}

class KBvExtractDecl internal constructor(
    ctx: KContext,
    val high: Int,
    val low: Int,
    value: KExpr<KBvSort>
) : KFuncDecl1<KBvSort, KBvSort>(
    ctx,
    name = "extract",
    resultSort = ctx.mkBvSort((high - low + 1).toUInt()),
    value.sort,
), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KBvSort>
    ): KApp<KBvSort, KBvSort> = mkBvExtractExprNoSimplify(high, low, arg)

    override val parameters: List<Any>
        get() = listOf(high, low)
}

class KSignExtDecl internal constructor(
    ctx: KContext,
    val i: Int,
    value: KBvSort
) : KFuncDecl1<KBvSort, KBvSort>(
    ctx,
    name = "sign_extend",
    resultSort = ctx.mkBvSort(value.sizeBits + i.toUInt()),
    value,
), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KBvSort>
    ): KApp<KBvSort, KBvSort> = mkBvSignExtensionExprNoSimplify(i, arg)

    override val parameters: List<Any>
        get() = listOf(i)
}

class KZeroExtDecl internal constructor(
    ctx: KContext,
    val i: Int,
    value: KBvSort
) : KFuncDecl1<KBvSort, KBvSort>(
    ctx,
    "zero_extend",
    resultSort = ctx.mkBvSort(value.sizeBits + i.toUInt()),
    value,
), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KBvSort>
    ): KApp<KBvSort, KBvSort> = mkBvZeroExtensionExprNoSimplify(i, arg)

    override val parameters: List<Any>
        get() = listOf(i)
}

class KBvRepeatDecl internal constructor(
    ctx: KContext,
    val i: Int,
    value: KBvSort
) : KFuncDecl1<KBvSort, KBvSort>(
    ctx,
    "repeat",
    resultSort = ctx.mkBvSort(value.sizeBits * i.toUInt()),
    value,
), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KBvSort>
    ): KApp<KBvSort, KBvSort> = mkBvRepeatExprNoSimplify(i, arg)

    override val parameters: List<Any>
        get() = listOf(i)
}

class KBvShiftLeftDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvshl", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<T>, arg1: KExpr<T>): KApp<T, *> = mkBvShiftLeftExprNoSimplify(arg0, arg1)
}

class KBvLogicalShiftRightDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvlshr", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<T, *> = mkBvLogicalShiftRightExprNoSimplify(arg0, arg1)
}

class KBvArithShiftRightDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "bvashr", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<T, *> = mkBvArithShiftRightExprNoSimplify(arg0, arg1)
}

class KBvRotateLeftDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(ctx, "ext_rotate_left", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<T, T> = mkBvRotateLeftExprNoSimplify(arg0, arg1)
}

class KBvRotateLeftIndexedDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    val i: Int,
    valueSort: T
) : KFuncDecl1<T, T>(ctx, "rotate_left", resultSort = valueSort, valueSort), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<T>
    ): KApp<T, T> = mkBvRotateLeftIndexedExprNoSimplify(i, arg)

    override val parameters: List<Any>
        get() = listOf(i)
}

class KBvRotateRightDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<T, T, T>(
    ctx,
    "ext_rotate_right",
    resultSort = arg0Sort,
    arg0Sort,
    arg1Sort
) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>
    ): KApp<T, T> = mkBvRotateRightExprNoSimplify(arg0, arg1)
}

class KBvRotateRightIndexedDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    val i: Int,
    valueSort: T
) : KFuncDecl1<T, T>(
    ctx,
    "rotate_right",
    resultSort = valueSort,
    valueSort
), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<T>
    ): KApp<T, T> = mkBvRotateRightIndexedExprNoSimplify(i, arg)

    override val parameters: List<Any>
        get() = listOf(i)
}

class KBv2IntDecl internal constructor(
    ctx: KContext,
    value: KBvSort,
    val isSigned: Boolean
) : KFuncDecl1<KIntSort, KBvSort>(
    ctx,
    "bv2int",
    resultSort = ctx.mkIntSort(),
    value
), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<KBvSort>
    ): KApp<KIntSort, KBvSort> = mkBv2IntExprNoSimplify(arg, isSigned)

    override val parameters: List<Any>
        get() = listOf(isSigned)
}

class KBvAddNoOverflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T,
    val isSigned: Boolean
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_add_no_overflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
), KParameterizedFuncDecl {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvAddNoOverflowExprNoSimplify(arg0, arg1, isSigned)

    override val parameters: List<Any>
        get() = listOf(isSigned)
}

class KBvAddNoUnderflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_add_no_underflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvAddNoUnderflowExprNoSimplify(arg0, arg1)
}

class KBvSubNoOverflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_sub_no_overflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvSubNoOverflowExprNoSimplify(arg0, arg1)
}

class KBvSubNoUnderflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T,
    val isSigned: Boolean
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_sub_no_underflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
), KParameterizedFuncDecl {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvSubNoUnderflowExprNoSimplify(arg0, arg1, isSigned)

    override val parameters: List<Any>
        get() = listOf(isSigned)
}

class KBvDivNoOverflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_div_no_overflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvDivNoOverflowExprNoSimplify(arg0, arg1)
}

class KBvNegNoOverflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    value: T
) : KFuncDecl1<KBoolSort, T>(
    ctx,
    "bv_neg_no_overflow",
    resultSort = ctx.mkBoolSort(),
    value
) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg: KExpr<T>
    ): KApp<KBoolSort, T> = mkBvNegationNoOverflowExprNoSimplify(arg)
}

class KBvMulNoOverflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T,
    val isSigned: Boolean
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_mul_no_overflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
), KParameterizedFuncDecl {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvMulNoOverflowExprNoSimplify(arg0, arg1, isSigned)

    override val parameters: List<Any>
        get() = listOf(isSigned)
}

class KBvMulNoUnderflowDecl<T : KBvSort> internal constructor(
    ctx: KContext,
    arg0Sort: T,
    arg1Sort: T
) : KFuncDecl2<KBoolSort, T, T>(
    ctx,
    "bv_mul_no_underflow",
    resultSort = ctx.mkBoolSort(),
    arg0Sort,
    arg1Sort
) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<T>,
        arg1: KExpr<T>,
    ): KApp<KBoolSort, *> = mkBvMulNoUnderflowExprNoSimplify(arg0, arg1)
}

private fun checkSortsAreTheSame(vararg sorts: KBvSort) {
    val sort = sorts.firstOrNull() ?: error("An empty array of sorts given for check")

    require(sorts.all { it == sort }) { "Given sorts are different: $sorts" }
}
