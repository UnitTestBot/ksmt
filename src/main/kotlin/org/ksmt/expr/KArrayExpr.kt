package org.ksmt.expr

import org.ksmt.decl.KDecl
import org.ksmt.sort.KSort

abstract class KArrayExpr<Domain : KExpr<Domain>, Range : KExpr<Range>> : KExpr<KArrayExpr<Domain, Range>>() {
    abstract override val sort: KSort<KArrayExpr<Domain, Range>>
    abstract override val decl: KDecl<KArrayExpr<Domain, Range>>
}

fun <Domain : KExpr<Domain>, Range : KExpr<Range>> KExpr<KArrayExpr<Domain, Range>>.store(index: KExpr<Domain>, value: KExpr<Range>) =
    mkArrayStore(this, index, value)

fun <Domain : KExpr<Domain>, Range : KExpr<Range>> KExpr<KArrayExpr<Domain, Range>>.select(index: KExpr<Domain>) =
    mkArraySelect(this, index)