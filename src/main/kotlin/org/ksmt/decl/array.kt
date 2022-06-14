package org.ksmt.decl

import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStoreDecl<D : KSort, R : KSort>(sort: KArraySort<D, R>) : KArrayDecl<D, R>("store", sort)

class KArraySelectDecl<R : KSort>(sort: R) : KDecl<R>("select", sort)
