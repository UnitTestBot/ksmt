package org.ksmt.expr

import org.ksmt.sort.KArithSort

abstract class KArithExpr<T : KArithSort<T>, A : KExpr<*>>(args: List<A>) : KApp<T, A>(args)
