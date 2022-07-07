package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KExtractExpr
import org.ksmt.expr.KRepeatExpr
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KIntSort

abstract class KBitVecValueDecl<T : KBvSort> internal constructor(ctx: KContext, val value: String, sort: T) :
    KConstDecl<T>(ctx, "#b$value", sort) {

    internal constructor(ctx: KContext, value: Number, sort: T) : this(ctx, value.toBinary(), sort)
}

class KBitVec8ValueDecl internal constructor(ctx: KContext, private val byteValue: Byte) :
    KBitVecValueDecl<KBv8Sort>(ctx, byteValue, ctx.mkBv8Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv8Sort, *> = ctx.mkBv(byteValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec16ValueDecl internal constructor(ctx: KContext, private val shortValue: Short) :
    KBitVecValueDecl<KBv16Sort>(ctx, shortValue, ctx.mkBv16Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv16Sort, *> = ctx.mkBv(shortValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec32ValueDecl internal constructor(ctx: KContext, private val intValue: Int) :
    KBitVecValueDecl<KBv32Sort>(ctx, intValue, ctx.mkBv32Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBv32Sort, *> = ctx.mkBv(intValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec64ValueDecl internal constructor(ctx: KContext, private val longValue: Long) :
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

class KBvRedAndDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvredand", ctx.bvSortWithSingleElement, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvRedAndExpr(arg)
}

class KBvRedOrDecl(ctx: KContext, valueSort: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "bvredor", ctx.bvSortWithSingleElement, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvRedOrExpr(arg)
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

class KBvNegDecl(ctx: KContext, valueSort: KBvSort) : KFuncDecl1<KBvSort, KBvSort>(ctx, "bvneg", valueSort, valueSort) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> = mkBvNegExpr(arg)
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

class KBvUDivDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvudiv", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvUDivExpr(arg0, arg1)
}

class KBvSDivDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsdiv", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSDivExpr(arg0, arg1)
}

class KBvURemDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvurem", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvURemExpr(arg0, arg1)
}

class KBvSRemDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsrem", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSRemExpr(arg0, arg1)
}

class KBvSModDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvsmod", left, right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSModExpr(arg0, arg1)
}

class KBvULTDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvult", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvULTExpr(arg0, arg1)
}

class KBvSLTDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvslt", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSLTExpr(arg0, arg1)
}

class KBvSLEDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsle", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSLEExpr(arg0, arg1)
}


class KBvULEDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvule", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvULEExpr(arg0, arg1)
}

class KBvUGEDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvuge", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvUGEExpr(arg0, arg1)
}

class KBvSGEDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvsge", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvSGEExpr(arg0, arg1)
}

class KBvUGTDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "bvugt", ctx.mkBoolSort(), right, left) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBoolSort, *> =
        mkBvUGTExpr(arg0, arg1)
}

class KBvSGTDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
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
        parameters = listOf(high, low)
    ) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkExtractExpr(parameters[0] as Int, parameters[1] as Int, arg)
}

class KSignExtDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "sign_extend",
        ctx.mkBvSort(value.sizeBits + i.toUInt()),
        value,
        parameters = listOf(i)
    ) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkSignExtExpr(parameters.single() as Int, arg)
}

class KZeroExtDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "zero_extend",
        ctx.mkBvSort(value.sizeBits + i.toUInt()),
        value,
        parameters = listOf(i)
    ) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkZeroExtExpr(parameters.single() as Int, arg)
}

class KRepeatDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(
        ctx,
        "repeat",
        ctx.mkBvSort(value.sizeBits * i.toUInt()),
        value,
        parameters = listOf(i)
    ) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkRepeatExpr(parameters.single() as Int, arg)
}

class KBvSHLDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvshl", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvSHLExpr(arg0, arg1)
}

class KBvLSHRDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvlshr", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvLSHRExpr(arg0, arg1)
}

class KBvASHRDecl(ctx: KContext, left: KBvSort, right: KBvSort) :
    KFuncDecl2<KBvSort, KBvSort, KBvSort>(ctx, "bvashr", left, left, right) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg0: KExpr<KBvSort>, arg1: KExpr<KBvSort>): KApp<KBvSort, *> = mkBvASHRExpr(arg0, arg1)
}

class KBvRotateLeftDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "rotate_left", value, value, parameters = listOf(i)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkBvRotateLeftExpr(parameters.single() as Int, arg)
}

class KBvRotateRightDecl(ctx: KContext, i: Int, value: KBvSort) :
    KFuncDecl1<KBvSort, KBvSort>(ctx, "rotate_right", value, value, parameters = listOf(i)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KBvSort, KExpr<KBvSort>> =
        mkBvRotateRightExpr(parameters.single() as Int, arg)
}

// TODO wrong declaration, should contain boolean parameter
// name??? looks like bv2int if it unsigned and false otherwise
class KBv2IntDecl(ctx: KContext, value: KBvSort) : KFuncDecl1<KIntSort, KBvSort>(ctx, "", ctx.mkIntSort(), value) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(arg: KExpr<KBvSort>): KApp<KIntSort, KExpr<KBvSort>> = mkBv2IntExpr(arg)
}

// TODO names??? = and => in z3
class KBvAddNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvAddNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)
}

class KBvAddNoUnderflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvAddNoUnderflowExpr(arg0, arg1, parameters.single() as Boolean)
}

class KBvSubNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvSubNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)
}

class KBvDivNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvDivNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)
}

class KBvNegNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvNegNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)
}

class KBvMulNoOverflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvMulNoOverflowExpr(arg0, arg1, parameters.single() as Boolean)
}

class KBvMulNoUnderflowDecl(ctx: KContext, left: KBvSort, right: KBvSort, isSigned: Boolean) :
    KFuncDecl2<KBoolSort, KBvSort, KBvSort>(ctx, "", ctx.mkBoolSort(), left, right, parameters = listOf(isSigned)) {
    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)

    override fun KContext.apply(
        arg0: KExpr<KBvSort>,
        arg1: KExpr<KBvSort>,
    ): KApp<KBoolSort, *> = mkBvMulNoUnderflowExpr(arg0, arg1, parameters.single() as Boolean)
}

private fun Number.toBinary(): String = when (this) {
    is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
    is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
    is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
    is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
    else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
}
