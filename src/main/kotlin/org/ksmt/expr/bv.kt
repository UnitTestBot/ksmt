package org.ksmt.expr

import org.ksmt.sort.KSort

interface KBVSize
interface KBVSize32 : KBVSize

class KBVSort<S : KBVSize> : KSort() {

}


fun <S : KBVSize> mkBVAdd(lhs: KExpr<KBVSort<S>>, rhs: KExpr<KBVSort<S>>) {
    TODO()
}

fun tets() {
}