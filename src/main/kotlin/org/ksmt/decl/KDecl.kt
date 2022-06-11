package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KDecl<T : KExpr<T>> {
    abstract val sort: KSort<T>
}
