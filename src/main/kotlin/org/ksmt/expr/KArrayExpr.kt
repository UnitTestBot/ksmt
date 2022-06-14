package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

abstract class KArrayExpr<D : KSort, R : KSort> : KExpr<KArraySort<D, R>>() {
    abstract override val sort: KArraySort<D, R>
    abstract override val decl: KDecl<KArraySort<D, R>>
}
