package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KDecl<T : KSort>(val name: String, val sort: T){
    abstract fun apply(args: List<KExpr<*>>): KExpr<T>
}
