package org.ksmt.decl

import org.ksmt.expr.KArrayExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

class KStoreArrayDecl<Domain : KExpr<Domain>, Range : KExpr<Range>>(sort: KSort<KArrayExpr<Domain, Range>>) :
    KArrayDecl<Domain, Range>(sort)

class KSelectArrayDecl<Range : KExpr<Range>>(override val sort: KSort<Range>) : KDecl<Range>()
