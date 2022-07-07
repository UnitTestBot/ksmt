package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KDecl
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBv16Sort
import org.ksmt.sort.KBv32Sort
import org.ksmt.sort.KBv64Sort
import org.ksmt.sort.KBv8Sort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KIntSort

abstract class KBitVecValue<T : KBvSort>(
    ctx: KContext
) : KApp<T, KExpr<*>>(ctx) {
    override val args: List<KExpr<*>> = emptyList()
}

abstract class KBitVecNumberValue<T : KBvSort, N : Number>(
    ctx: KContext,
    val numberValue: N
) : KBitVecValue<T>(ctx)

class KBitVec8Value internal constructor(ctx: KContext, byteValue: Byte) :
    KBitVecNumberValue<KBv8Sort, Byte>(ctx, byteValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv8Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv8Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBv8Sort = ctx.mkBv8Sort()
}

class KBitVec16Value internal constructor(ctx: KContext, shortValue: Short) :
    KBitVecNumberValue<KBv16Sort, Short>(ctx, shortValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv16Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv16Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBv16Sort = ctx.mkBv16Sort()
}

class KBitVec32Value internal constructor(ctx: KContext, intValue: Int) :
    KBitVecNumberValue<KBv32Sort, Int>(ctx, intValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv32Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv32Sort> = ctx.mkBvDecl(numberValue)

    override fun sort(): KBv32Sort = ctx.mkBv32Sort()
}

class KBitVec64Value internal constructor(ctx: KContext, longValue: Long) :
    KBitVecNumberValue<KBv64Sort, Long>(ctx, longValue) {
    override fun accept(transformer: KTransformer): KExpr<KBv64Sort> = transformer.transform(this)

    override fun decl(): KDecl<KBv64Sort> = ctx.mkBvDecl(numberValue)
    override fun sort(): KBv64Sort = ctx.mkBv64Sort()
}

class KBitVecCustomValue internal constructor(
    ctx: KContext,
    val value: String,
    private val sizeBits: UInt
) : KBitVecValue<KBvSort>(ctx) {

    init {
        require(value.length.toUInt() == sizeBits) {
            "Provided string $value consist of ${value.length} symbols, but $sizeBits were expected"
        }
    }

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

    override fun decl(): KDecl<KBvSort> = ctx.mkBvDecl(value, sizeBits)

    override fun sort(): KBvSort = ctx.mkBvSort(sizeBits)
}

// expressions for operations
/**
 * Bitwise negation.
 */
class KBvNotExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNotDecl(value.sort())

    override fun sort(): KBvSort = value.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

/**
 * Takes conjunction of bits in a vector, return vector of length 1.
 */
class KBvRedAndExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRedAndDecl(value.sort())

    override fun sort(): KBvSort = ctx.bvSortWithSingleElement

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Take disjunction of bits in a vector, return vector of length 1.
 */
class KBvRedOrExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRedOrDecl(value.sort())

    override fun sort(): KBvSort = ctx.bvSortWithSingleElement

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise conjunction.
 */
class KBvAndExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvAndDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise disjunction.
 */
class KBvOrExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvOrDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()
    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise XOR.
 */
class KBvXorExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvXorDecl(left.sort(), right.sort())
    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise NAND.
 */
class KBvNAndExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNAndDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise NOR.
 */
class KBvNorExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNorDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Bitwise XNOR.
 */
class KBvXNorExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvXNorDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Standard two's complement unary minus.
 */
class KBvNegExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvNegDecl(value.sort())
    override fun sort(): KBvSort = value.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement addition.
 */
class KBvAddExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvAddDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement subtraction.
 */
class KBvSubExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSubDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Two's complement multiplication.
 */
class KBvMulExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvMulDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Unsigned division.
 *
 * It is defined as the floor of `t1 / t2` if `t2` is different from zero.
 * Otherwise, the result is undefined.
 */
class KBvUDivExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvUDivDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Signed division.
 *
 * It is defined as:
 * * the floor of the `t1 / t2` if `t2` is different from zero and `t1 * t2 >= 0`
 * * the ceiling of `t1 / t2` if `t2` if different from zero and `t1 * t2 < 0`
 * * if `t2` is zero, then the result is undefined.
 */
class KBvSDivExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSDivDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

/**
 * Unsigned remainder.
 *
 * It is defined as `t1 - (t1 /u t2) * t2`, where `\u` represents unsigned division.
 * If `t2` is zero, then the result is undefined.
 */
class KBvURemExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvURemDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

class KBvSRemExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSRemDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

class KBvSModExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSModDecl(left.sort(), right.sort())
    override fun sort(): KBvSort = left.sort()


    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

class KBvULTExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvULTDecl(left.sort(), right.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSLTExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSLTDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvULEExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvULEDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()


    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSLEExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSLEDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()


    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvUGEExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUGEDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSGEExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSGEDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvUGTExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvUGTDecl(left.sort(), right.sort())

    override fun sort(): KBoolSort = ctx.mkBoolSort()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KBvSGTExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBoolSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBoolSort> = ctx.mkBvSGTDecl(left.sort(), right.sort())
    override fun sort(): KBoolSort = ctx.mkBoolSort()

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KConcatExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkConcatDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = decl().sort
    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)
}

class KExtractExpr internal constructor(
    ctx: KContext,
    private val high: Int,
    private val low: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkExtractDecl(high, low, value)

    override fun sort(): KBvSort = ctx.mkBvSort((high - low + 1).toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = TODO()
}

class KSignExtExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkSignExtDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits + i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KZeroExtExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkZeroExtDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits + i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KRepeatExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkRepeatDecl(i, value.sort())

    override fun sort(): KBvSort = ctx.mkBvSort(value.sort().sizeBits * i.toUInt())

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KBvSHLExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvSHLDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KBvLSHRExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvLSHRDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}


class KBvASHRExpr internal constructor(
    ctx: KContext,
    val left: KExpr<KBvSort>,
    val right: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(left, right)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvASHRDecl(left.sort(), right.sort())

    override fun sort(): KBvSort = left.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}


class KBvRotateLeftExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRotateLeftDecl(i, value.sort())

    override fun sort(): KBvSort = value.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KBvRotateRightExpr internal constructor(
    ctx: KContext,
    val i: Int,
    val value: KExpr<KBvSort>
) : KApp<KBvSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KBvSort> = ctx.mkBvRotateRightDecl(i, value.sort())

    override fun sort(): KBvSort = value.sort()

    override fun accept(transformer: KTransformer): KExpr<KBvSort> = transformer.transform(this)

}

class KBv2IntExpr internal constructor(
    ctx: KContext,
    val value: KExpr<KBvSort>
) : KApp<KIntSort, KExpr<KBvSort>>(ctx) {
    override val args: List<KExpr<KBvSort>> by lazy {
        listOf(value)
    }

    override fun decl(): KDecl<KIntSort> = ctx.mkBv2IntDecl(value.sort())

    override fun sort(): KIntSort = ctx.mkIntSort()

    override fun accept(transformer: KTransformer): KExpr<KIntSort> = transformer.transform(this)

}

// TODO think about their declarations -- in z3 they have decl: declare-fun => (Bool Bool) Bool
// toString for `mkBVAddNoOverflow(mkBV(2, 2), mkBV(2, 2), true)`
// (=> (and (bvslt #b00 #b10) (bvslt #b00 #b10)) (bvslt #b00 (bvadd #b10 #b10)))
class KBvAddNoOverflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvAddNoOverflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)

class KBvAddNoUnderflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvAddNoUnderflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)

class KBvSubNoOverflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvSubNoOverflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)

class KBvDivNoOverflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvDivNoOverflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)

class KBvNegNoOverflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvNegNoOverflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)

class KBvMulNoOverflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvMulNoOverflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)

class KBvMulNoUnderflowExpr internal constructor(
    ctx: KContext,
    left: KExpr<KBvSort>,
    right: KExpr<KBvSort>,
    isSigned: Boolean
) : KFunctionApp<KBoolSort>(
    ctx,
    ctx.mkBvMulNoUnderflowDecl(left.sort(), right.sort(), isSigned),
    listOf(left, right, if (isSigned) ctx.trueExpr else ctx.falseExpr)
)



