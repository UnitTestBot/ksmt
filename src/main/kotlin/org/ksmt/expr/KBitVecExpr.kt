package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KBVSort

interface KBVSize
object KBVSize8 : KBVSize
object KBVSize16 : KBVSize
object KBVSize32 : KBVSize
object KBVSize64 : KBVSize

class KBVCustomSize(val sizeBits: UInt) : KBVSize

class BitVecExpr<T : KBVSort<KBVSize>>(
    decl: KDecl<T>,
    args: List<KExpr<*>>
) : KApp<T, KExpr<*>>(decl, args) {
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

fun <S : KBVSize> mkBVAdd(lhs: KExpr<KBVSort<S>>, rhs: KExpr<KBVSort<S>>) {
    TODO()
}
