package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv1Sort
import org.ksmt.sort.KIntSort

abstract class KBitVecValueDecl<T : KBvSort> internal constructor(ctx: KContext, val value: String, sort: T) :
    KConstDecl<T>(ctx, "#b$value", sort) {

    internal constructor(ctx: KContext, value: Number, sort: T) : this(ctx, value.toBinary(), sort)
}

class KBitVec1ValueDecl internal constructor(ctx: KContext, private val boolValue: Boolean) :
    KBitVecValueDecl<KBv1Sort>(ctx, if (boolValue) "1" else "0", ctx.mkBv1Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv1Sort, *> = ctx.mkBv(boolValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec8ValueDecl internal constructor(ctx: KContext, val byteValue: Byte) :
    KBitVecValueDecl<KBv8Sort>(ctx, byteValue, ctx.mkBv8Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv8Sort, *> = ctx.mkBv(byteValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec16ValueDecl internal constructor(ctx: KContext, val shortValue: Short) :
    KBitVecValueDecl<KBv16Sort>(ctx, shortValue, ctx.mkBv16Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv16Sort, *> = ctx.mkBv(shortValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec32ValueDecl internal constructor(ctx: KContext, val intValue: Int) :
    KBitVecValueDecl<KBv32Sort>(ctx, intValue, ctx.mkBv32Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv32Sort, *> = ctx.mkBv(intValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec64ValueDecl internal constructor(ctx: KContext, val longValue: Long) :
    KBitVecValueDecl<KBv64Sort>(ctx, longValue, ctx.mkBv64Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv64Sort, *> = ctx.mkBv(longValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVecCustomSizeValueDecl internal constructor(
    ctx: KContext,
    value: String,
    sizeBits: UInt
) : KBitVecValueDecl<KBvSort>(ctx, value, ctx.mkBvSort(sizeBits)) {
    override fun apply(args: List<KExpr<*>>): KApp<KBvSort, *> = ctx.mkBv(value, sort.sizeBits)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}


// Expressions with bit-vectors
class KBvNotDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvnot", resultSort = valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvNotExpr(arg)
}

class KBvReductionAndDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvredand", resultSort = ctx.mkBv1Sort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvReductionAndExpr(arg)
}

class KBvReductionOrDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvredor", resultSort = ctx.mkBv1Sort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvReductionOrExpr(arg)
}

class KBvAndDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvand", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvAndExpr(arg0, arg1)
}

class KBvOrDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvOrExpr(arg0, arg1)
}

class KBvXorDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvxor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvXorExpr(arg0, arg1)
}

class KBvNAndDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvnand", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvNAndExpr(arg0, arg1)
}

class KBvNorDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvnor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvNorExpr(arg0, arg1)
}

class KBvXNorDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvxnor", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvXNorExpr(arg0, arg1)
}

class KBvNegationDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvneg", resultSort = valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvNegationExpr(arg)
}

class KBvAddDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvadd", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvAddExpr(arg0, arg1)
}

class KBvSubDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsub", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSubExpr(arg0, arg1)
}

class KBvMulDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvmul", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvMulExpr(arg0, arg1)
}

class KBvUnsignedDivDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvudiv", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvUnsignedDivExpr(arg0, arg1)
}

class KBvSignedDivDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsdiv", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvSignedDivExpr(arg0, arg1)
}

class KBvUnsignedRemDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvurem", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvUnsignedRemExpr(arg0, arg1)
}

class KBvSignedRemDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsrem", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvSignedRemExpr(arg0, arg1)
}

class KBvSignedModDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsmod", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvSignedModExpr(arg0, arg1)
}

class KBvUnsignedLessDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvult", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvUnsignedLessExpr(arg0, arg1)
}

class KBvSignedLessDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvslt", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvSignedLessExpr(arg0, arg1)
}

class KBvSignedLessOrEqualDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsle", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvSignedLessOrEqualExpr(arg0, arg1)
}


class KBvUnsignedLessOrEqualDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvule", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvUnsignedLessOrEqualExpr(arg0, arg1)
}

class KBvUnsignedGreaterOrEqualDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvuge", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvUnsignedGreaterOrEqualExpr(arg0, arg1)
}

class KBvSignedGreaterOrEqualDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsge", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvSignedGreaterOrEqualExpr(arg0, arg1)
}

class KBvUnsignedGreaterDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvugt", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvUnsignedGreaterExpr(arg0, arg1)
}

class KBvSignedGreaterDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsgt", resultSort = ctx.mkBoolSort(), arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBoolSort, *> = mkBvSignedGreaterExpr(arg0, arg1)
}

class KConcatDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(
        ctx,
        "concat",
        resultSort = ctx.mkBvSort(arg0Sort.sizeBits + arg1Sort.sizeBits),
        arg0Sort,
        arg1Sort
    ) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkConcatExpr(arg0, arg1)
}

class KExtractDecl(ctx: KContext, high: Int, low: Int, value: KExpr<KBvSort>) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "extract",
        resultSort = ctx.mkBvSort((high - low + 1).toUInt()),
        value.sort(),
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkExtractExpr(parameters[0] as Int, parameters[1] as Int, arg)

    override val parameters: List<Any> = listOf(high, low)
}

class KSignExtDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "sign_extend",
        resultSort = ctx.mkBvSort(value.sizeBits + i.toUInt()),
        value,
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkSignExtExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> = listOf(i)
}

class KZeroExtDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "zero_extend",
        resultSort = ctx.mkBvSort(value.sizeBits + i.toUInt()),
        value,
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkZeroExtExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> = listOf(i)
}

class KRepeatDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "repeat",
        resultSort = ctx.mkBvSort(value.sizeBits * i.toUInt()),
        value,
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkRepeatExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> = listOf(i)
}

class KBvShiftLeftDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvshl", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvShiftLeftExpr(arg0, arg1)
}

class KBvLogicalShiftRightDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvlshr", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvLogicalShiftRightExpr(arg0, arg1)
}

class KBvArithShiftRightDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvashr", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, *> = mkBvArithShiftRightExpr(arg0, arg1)
}

class KBvRotateLeftDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "ext_rotate_left", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, KExpr<KBvSort>> = mkBvRotateLeftExpr(arg0, arg1)
}

class KBvRotateLeftIndexedDecl(ctx: KContext, i: Int, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "rotate_left", resultSort = valueSort, valueSort), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkBvRotateLeftIndexedExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> = listOf(i)
}

class KBvRotateRightDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "ext_rotate_right", resultSort = arg0Sort, arg0Sort, arg1Sort) {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>
    ): KApp<KBvSort, KExpr<KBvSort>> = mkBvRotateRightExpr(arg0, arg1)
}

class KBvRotateRightIndexedDecl(ctx: KContext, i: Int, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "rotate_right", resultSort = valueSort, valueSort), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkBvRotateRightIndexedExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> = listOf(i)
}

class KBv2IntDecl(ctx: KContext, value: KBvSort, isSigned: Boolean) :
    KFuncDecl1<KIntSort, KBvSort>(ctx, "bv2int", resultSort = ctx.mkIntSort(), value), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KIntSort, KExpr<KBvSort>> =
        mkBv2IntExpr(arg, parameters.single() as Boolean)

    override val parameters: List<Any> = listOf(isSigned)
}

class KBvAddNoOverflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
        ctx,
        "bv_add_no_overflow",
        resultSort = ctx.mkBoolSort(),
        arg0Sort,
        arg1Sort
    ),
    KParameterizedFuncDecl {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvAddNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)

    override val parameters: List<Any> = listOf(isSigned)
}

class KBvAddNoUnderflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
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
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvAddNoUnderflowExpr(arg0, arg1)
}

class KBvSubNoOverflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
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
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvSubNoOverflowExpr(arg0, arg1)
}

class KBvSubNoUnderflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
        ctx,
        "bv_sub_no_underflow",
        resultSort = ctx.mkBoolSort(),
        arg0Sort,
        arg1Sort
    ),
    KParameterizedFuncDecl {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvSubNoUnderflowExpr(arg0, arg1, parameters.single() as Boolean)

    override val parameters: List<Any> = listOf(isSigned)
}

class KBvDivNoOverflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
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
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvDivNoOverflowExpr(arg0, arg1)
}

class KBvNegNoOverflowDecl(ctx: KContext, value: KBvSort) :
    KFuncDecl1<KBoolSort, KBvSort>(ctx, "bv_neg_no_overflow", resultSort = ctx.mkBoolSort(), value) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBoolSort, KExpr<KBvSort>> =
        mkBvNegationNoOverflowExpr(arg)
}

class KBvMulNoOverflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
        ctx,
        "bv_mul_no_overflow",
        resultSort = ctx.mkBoolSort(),
        arg0Sort,
        arg1Sort
    ),
    KParameterizedFuncDecl {
    init {
        checkSortsAreTheSame(arg0Sort, arg1Sort)
    }

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvMulNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)

    override val parameters: List<Any> = listOf(isSigned)
}

class KBvMulNoUnderflowDecl(ctx: KContext, arg0Sort: KBvSort, arg1Sort: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(
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
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvMulNoUnderflowExpr(arg0, arg1)
}

// We can have here `0` as a pad symbol since `toString` can return a string
// containing fewer symbols than `sizeBits` only for non-negative numbers
internal fun Number.toBinary(): String = when (this) {
    is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
    is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
    is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
    is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
    else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
}

private fun checkSortsAreTheSame(vararg sorts: KBvSort) {
    val sort = sorts.firstOrNull() ?: error("An empty array of sorts given for check")

    require(sorts.all { it === sort }) { "Given sorts are different: $sorts" }
}