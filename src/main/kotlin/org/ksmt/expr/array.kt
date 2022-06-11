package org.ksmt.expr

import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KSelectArrayDecl
import org.ksmt.decl.KStoreArrayDecl
import org.ksmt.sort.range

class KArrayStore<Domain : KExpr<Domain>, Range : KExpr<Range>> internal constructor(
    val array: KExpr<KArrayExpr<Domain, Range>>,
    val index: KExpr<Domain>,
    val value: KExpr<Range>
) : KArrayExpr<Domain, Range>() {
    override val sort = array.sort
    override val decl = KStoreArrayDecl(array.sort)
    override val args = listOf(array, index, value)
}

class KArraySelect<Domain : KExpr<Domain>, Range : KExpr<Range>> internal constructor(
    val array: KExpr<KArrayExpr<Domain, Range>>,
    val index: KExpr<Domain>
) : KExpr<Range>() {
    override val sort = array.sort.range
    override val decl = KSelectArrayDecl(array.sort.range)
    override val args = listOf(array, index)
}

class KArrayConst<Domain : KExpr<Domain>, Range : KExpr<Range>>(override val decl: KConstDecl<KArrayExpr<Domain, Range>>) :
    KArrayExpr<Domain, Range>() {
    override val sort = decl.sort
    override val args = emptyList<KExpr<*>>()
}

fun <Domain : KExpr<Domain>, Range : KExpr<Range>> mkArrayStore(
    array: KExpr<KArrayExpr<Domain, Range>>,
    index: KExpr<Domain>,
    value: KExpr<Range>
) = KArrayStore(array, index, value)

fun <Domain : KExpr<Domain>, Range : KExpr<Range>> mkArraySelect(
    array: KExpr<KArrayExpr<Domain, Range>>,
    index: KExpr<Domain>
) = KArraySelect(array, index)


fun <Domain : KExpr<Domain>, Range : KExpr<Range>> mkArrayConst(decl: KConstDecl<KArrayExpr<Domain, Range>>) =
    KArrayConst(decl)
