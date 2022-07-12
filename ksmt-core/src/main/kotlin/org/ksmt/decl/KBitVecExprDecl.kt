package org.ksmt.decl

import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.sort.KBV16Sort
import org.ksmt.sort.KBV1Sort
import org.ksmt.sort.KBV32Sort
import org.ksmt.sort.KBV64Sort
import org.ksmt.sort.KBV8Sort
import org.ksmt.sort.KBVSort

abstract class KBitVecExprDecl<T : KBVSort> internal constructor(ctx: KContext, val value: String, sort: T) :
    KConstDecl<T>(ctx, "#b$value", sort) {

    internal constructor(ctx: KContext, value: Number, sort: T) : this(ctx, value.toBinary(), sort)
}

class KBitVec1ExprDecl internal constructor(ctx: KContext, private val boolValue: Boolean) :
    KBitVecExprDecl<KBV1Sort>(ctx, if (boolValue) "1" else "0", ctx.mkBv1Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBV1Sort, *> = ctx.mkBV(boolValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec8ExprDecl internal constructor(ctx: KContext, private val byteValue: Byte) :
    KBitVecExprDecl<KBV8Sort>(ctx, byteValue, ctx.mkBv8Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBV8Sort, *> = ctx.mkBV(byteValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec16ExprDecl internal constructor(ctx: KContext, private val shortValue: Short) :
    KBitVecExprDecl<KBV16Sort>(ctx, shortValue, ctx.mkBv16Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBV16Sort, *> = ctx.mkBV(shortValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec32ExprDecl internal constructor(ctx: KContext, private val intValue: Int) :
    KBitVecExprDecl<KBV32Sort>(ctx, intValue, ctx.mkBv32Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBV32Sort, *> = ctx.mkBV(intValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVec64ExprDecl internal constructor(ctx: KContext, private val longValue: Long) :
    KBitVecExprDecl<KBV64Sort>(ctx, longValue, ctx.mkBv64Sort()) {
    override fun apply(args: List<KExpr<*>>): KApp<KBV64Sort, *> = ctx.mkBV(longValue)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

class KBitVecCustomSizeExprDecl internal constructor(
    ctx: KContext,
    value: String,
    sizeBits: UInt
) : KBitVecExprDecl<KBVSort>(ctx, value, ctx.mkBvSort(sizeBits)) {
    override fun apply(args: List<KExpr<*>>): KApp<KBVSort, *> = ctx.mkBV(value, sort.sizeBits)

    override fun <R> accept(visitor: KDeclVisitor<R>): R = visitor.visit(this)
}

private fun Number.toBinary(): String = when (this) {
    is Byte -> toUByte().toString(radix = 2).padStart(Byte.SIZE_BITS, '0')
    is Short -> toUShort().toString(radix = 2).padStart(Short.SIZE_BITS, '0')
    is Int -> toUInt().toString(radix = 2).padStart(Int.SIZE_BITS, '0')
    is Long -> toULong().toString(radix = 2).padStart(Long.SIZE_BITS, '0')
    else -> error("Unsupported type for transformation into a binary string: ${this::class.simpleName}")
}
