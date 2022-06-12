package org.ksmt.decl

import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

abstract class KArrayDecl<D : KSort<D>, R : KSort<R>>(
    override val sort: KArraySort<D, R>
) : KDecl<KArraySort<D, R>>()
