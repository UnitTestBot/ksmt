package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort

abstract class KExpr<T : KExpr<T>> {
    abstract val sort: KSort<T>
    abstract val decl: KDecl<T>
    abstract val args: List<KExpr<*>>
}

infix fun <T : KExpr<T>> KExpr<T>.eq(other: KExpr<T>) = mkEq(this, other)
