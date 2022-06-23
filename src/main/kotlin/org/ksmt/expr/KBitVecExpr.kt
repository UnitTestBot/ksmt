package org.ksmt.expr

import org.ksmt.sort.KBVSort

interface KBVSize
object KBVSize8 : KBVSize
object KBVSize16 : KBVSize
object KBVSize32 : KBVSize
object KBVSize64 : KBVSize

class KBVCustomSize(val sizeBits: UInt) : KBVSize

abstract class BitVecExpr<T : KBVSort<KBVSize>>(
    override val args: List<KExpr<*>>
) : KApp<T, KExpr<*>> {
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

fun <S : KBVSize> mkBVAdd(lhs: KExpr<KBVSort<S>>, rhs: KExpr<KBVSort<S>>) {
    TODO()
}
