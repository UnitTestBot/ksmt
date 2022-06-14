package org.ksmt.decl

import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStoreDecl<D : KSort, R : KSort>(sort: KArraySort<D, R>) : KArrayDecl<D, R>(sort)

class KArraySelectDecl<R : KSort>(override val sort: R) : KDecl<R>()
