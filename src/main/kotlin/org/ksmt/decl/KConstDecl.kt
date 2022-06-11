package org.ksmt.decl

import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

class KConstDecl<T : KExpr<T>>(val name: String, override val sort: KSort<T>) : KDecl<T>()
