package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KArithSort

abstract class KArithExpr<T : KArithSort<T>, A : KExpr<*>>(
    decl: KDecl<T>,
    args: List<A>
) : KApp<T, A>(decl, args)
