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
class KBvNotDecl(ctx: KContext, valueSort: KBvSort) : KFuncDecl1<KBvSort, KBvSort>(ctx, "bvnot", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvNotExpr(arg)
}

class KBvReductionAndDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvredand", ctx.mkBv1Sort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvReductionAndExpr(arg)
}

class KBvReductionOrDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvredor", ctx.mkBv1Sort(), valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvReductionOrExpr(arg)
}

class KBvAndDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvand", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvAndExpr(arg0, arg1)
}

class KBvOrDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvor", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvOrExpr(arg0, arg1)
}

class KBvXorDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvxor", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvXorExpr(arg0, arg1)
}

class KBvNAndDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvnand", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvNAndExpr(arg0, arg1)
}

class KBvNorDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvnor", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvNorExpr(arg0, arg1)
}

class KBvXNorDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvxnor", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvXNorExpr(arg0, arg1)
}

class KBvNegationDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvneg", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvNegationExpr(arg)
}

class KBvAddDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvadd", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvAddExpr(arg0, arg1)
}

class KBvSubDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsub", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSubExpr(arg0, arg1)
}

class KBvMulDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvmul", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvMulExpr(arg0, arg1)
}

class KBvUnsignedDivDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvudiv", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> =
        mkBvUnsignedDivExpr(arg0, arg1)
}

class KBvSignedDivDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsdiv", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> =
        mkBvSignedDivExpr(arg0, arg1)
}

class KBvUnsignedRemDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvurem", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> =
        mkBvUnsignedRemExpr(arg0, arg1)
}

class KBvSignedRemDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsrem", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> =
        mkBvSignedRemExpr(arg0, arg1)
}

class KBvSignedModDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsmod", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> =
        mkBvSignedModExpr(arg0, arg1)
}

class KBvUnsignedLessDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvult", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvUnsignedLessExpr(arg0, arg1)
}

class KBvSignedLessDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvslt", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSignedLessExpr(arg0, arg1)
}

class KBvSignedLessOrEqualDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsle", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSignedLessOrEqualExpr(arg0, arg1)
}


class KBvUnsignedLessOrEqualDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvule", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvULEExpr(arg0, arg1)
}

class KBvUnsignedGreaterOrEqualDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvuge", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvUGEExpr(arg0, arg1)
}

class KBvSignedGreaterOrEqualDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsge", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSGEExpr(arg0, arg1)
}

class KBvUnsignedGreaterDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvugt", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvUGTExpr(arg0, arg1)
}

class KBvSignedGreaterDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsgt", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSGTExpr(arg0, arg1)
}

class KConcatDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "concat", ctx.mkBvSort(left.sizeBits + right.sizeBits), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkConcatExpr(arg0, arg1)
}

class KExtractDecl(ctx: KContext, high: Int, low: Int, value: KExpr<KBvSort>) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "extract",
        ctx.mkBvSort((high - low + 1).toUInt()),
        value.sort(),
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkExtractExpr(parameters[0] as Int, parameters[1] as Int, arg)

    override val parameters: List<Any> by lazy {
        listOf(high, low)
    }
}

class KSignExtDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "sign_extend",
        ctx.mkBvSort(value.sizeBits + i.toUInt()),
        value,
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkSignExtExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> by lazy {
        listOf(i)
    }
}

class KZeroExtDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "zero_extend",
        ctx.mkBvSort(value.sizeBits + i.toUInt()),
        value,
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkZeroExtExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> by lazy {
        listOf(i)
    }
}

class KRepeatDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "repeat",
        ctx.mkBvSort(value.sizeBits * i.toUInt()),
        value,
    ), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkRepeatExpr(parameters.single() as Int, arg)

    override val parameters: List<Any> by lazy {
        listOf(i)
    }
}

class KBvShiftLeftDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvshl", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSHLExpr(arg0, arg1)
}

class KBvLogicalShiftRightDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvlshr", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvLSHRExpr(arg0, arg1)
}

class KBvArithShiftRightDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvashr", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvASHRExpr(arg0, arg1)
}

class KBvRotateLeftDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "rotate_left", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkBvRotateLeftExpr(arg0, arg1)
}

class KBvRotateRightDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "rotate_right", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkBvRotateRightExpr(arg0, arg1)
}

// name??? looks like bv2int if it unsigned and false otherwise
class KBv2IntDecl(ctx: KContext, value: KBvSort, isSigned: Boolean) :
    KFuncDecl1<KIntSort, KBvSort>(ctx, "bv2int", ctx.mkIntSort(), value), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KIntSort, KExpr<KBvSort>> =
        mkBv2IntExpr(arg, parameters.single() as Boolean)

    override val parameters: List<Any> by lazy {
        listOf(isSigned)
    }
}

// TODO names??? = and => in z3
class KBvAddNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_add_no_overflow", ctx.mkBoolSort(), left, right), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvAddNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)

    override val parameters: List<Any> by lazy {
        listOf(isSigned)
    }
}

class KBvAddNoUnderflowDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_add_no_underflow", ctx.mkBoolSort(), left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvAddNoUnderflowExpr(arg0, arg1)
}

class KBvSubNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_sub_no_overflow", ctx.mkBoolSort(), left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvSubNoOverflowExpr(arg0, arg1)
}

class KBvSubNoUnderflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_sub_no_underflow", ctx.mkBoolSort(), left, right), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvSubNoUnderflowExpr(arg0, arg1, parameters.single() as Boolean)

    override val parameters: List<Any> by lazy {
        listOf(isSigned)
    }
}

class KBvDivNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_div_no_overflow", ctx.mkBoolSort(), left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvDivNoOverflowExpr(arg0, arg1)
}

class KBvNegNoOverflowDecl(ctx: KContext, value: KBvSort) :
    KFuncDecl1<KBoolSort, KBvSort>(ctx, "bv_neg_no_overflow", ctx.mkBoolSort(), value) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBoolSort, KExpr<KBvSort>> =
        mkBvNegationNoOverflowExpr(arg)
}

class KBvMulNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_mul_no_overflow", ctx.mkBoolSort(), left, right), KParameterizedFuncDecl {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvMulNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)

    override val parameters: List<Any> by lazy {
        listOf(isSigned)
    }
}

class KBvMulNoUnderflowDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bv_mul_no_underflow", ctx.mkBoolSort(), left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvMulNoUnderflowExpr(arg0, arg1)
}

internal fun Number.toBinary(): String = when (this) {
    is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
    is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
    is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
    is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
    else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
}
