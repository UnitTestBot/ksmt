package org.ksmt.decl

import org.ksmt.expr.KArrayExpr
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

abstract class KArrayDecl<Domain : KExpr<Domain>, Range : KExpr<Range>>(
    override val sort: KSort<KArrayExpr<Domain, Range>>
) : KDecl<KArrayExpr<Domain, Range>>()
