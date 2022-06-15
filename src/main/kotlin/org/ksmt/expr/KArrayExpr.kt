package org.ksmt.expr

import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

abstract class KArrayExpr<D : KSort, R : KSort, A : KExpr<*>>(args: List<A>) : KApp<KArraySort<D, R>, A>(args)
