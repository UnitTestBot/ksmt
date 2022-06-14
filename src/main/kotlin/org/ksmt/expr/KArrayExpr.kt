package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

abstract class KArrayExpr<D : KSort, R : KSort>(
    decl: KDecl<KArraySort<D, R>>,
    args: List<KExpr<*>>
) : KApp<KArraySort<D, R>, KExpr<*>>(decl, args)
