package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBV16Sort
import org.ksmt.sort.KBV1Sort
import org.ksmt.sort.KBV32Sort
import org.ksmt.sort.KBV64Sort
import org.ksmt.sort.KBV8Sort
import org.ksmt.sort.KBVSort

abstract class KBitVecExpr<T : KBVSort>(
    ctx: KContext
) : KApp<T, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>> = emptyList()
}

class KBitVec1Expr internal constructor(ctx: KContext, val value: Boolean) : KBitVecExpr<KBV1Sort>(ctx) {
    override fun accept(transformer: KTransformer): KExpr<KBV1Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBV1Sort> = ctx.mkBvDecl(value)

    override fun sort(): KBV1Sort = ctx.mkBv1Sort()
}

abstract class KBitVecNumberExpr<T : KBVSort, N : Number>(
    ctx: KContext,
    val numberValue: N
) : KBitVecExpr<T>(ctx)

class KBitVec8Expr internal constructor(ctx: KContext, byteValue: Byte) :
    KBitVecNumberExpr<KBV8Sort, Byte>(ctx, byteValue) {
    override fun accept(transformer: KTransformer): KExpr<KBV8Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBV8Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBV8Sort = ctx.mkBv8Sort()
}

class KBitVec16Expr internal constructor(ctx: KContext, shortValue: Short) :
    KBitVecNumberExpr<KBV16Sort, Short>(ctx, shortValue) {
    override fun accept(transformer: KTransformer): KExpr<KBV16Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBV16Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBV16Sort = ctx.mkBv16Sort()
}

class KBitVec32Expr internal constructor(ctx: KContext, intValue: Int) :
    KBitVecNumberExpr<KBV32Sort, Int>(ctx, intValue) {
    override fun accept(transformer: KTransformer): KExpr<KBV32Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBV32Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBV32Sort = ctx.mkBv32Sort()
}

class KBitVec64Expr internal constructor(ctx: KContext, longValue: Long) :
    KBitVecNumberExpr<KBV64Sort, Long>(ctx, longValue) {
    override fun accept(transformer: KTransformer): KExpr<KBV64Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBV64Sort> = ctx.mkBvDecl(numberValue)
    override fun sort(): KBV64Sort = ctx.mkBv64Sort()
}

class KBitVecCustomExpr internal constructor(
    ctx: KContext,
    val value: String,
    private val sizeBits: UInt
) : KBitVecExpr<KBVSort>(ctx) {

    init {
        require(value.length.toUInt() == sizeBits) {
            "Provided string $value consist of ${value.length} symbols, but $sizeBits were expected"
        }
    }

    override fun accept(transformer: KTransformer): KExpr<KBVSort> = transformer.transform(this)

    override fun decl(): KDecl<KBVSort> = ctx.mkBvDecl(value, sizeBits)

    override fun sort(): KBVSort = ctx.mkBvSort(sizeBits)
}
