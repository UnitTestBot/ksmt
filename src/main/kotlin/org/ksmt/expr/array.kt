package org.ksmt.expr

import org.ksmt.decl.KArraySelectDecl
import org.ksmt.decl.KArrayStoreDecl
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStore<D : KSort, R : KSort> internal constructor(
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    val value: KExpr<R>
) : KArrayExpr<D, R>() {
    override val sort = array.sort
    override val decl = KArrayStoreDecl(array.sort)
    override val args = listOf(array, index, value)
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KExpr<R>() {
    override val sort = array.sort.range
    override val decl = KArraySelectDecl(array.sort.range)
    override val args = listOf(array, index)
}

class KArrayConst<D : KSort, R : KSort>(override val decl: KConstDecl<KArraySort<D, R>>) :
    KArrayExpr<D, R>() {
    override val sort = decl.sort
    override val args = emptyList<KExpr<*>>()
}

fun <D : KSort, R : KSort> mkArrayStore(array: KExpr<KArraySort<D, R>>, index: KExpr<D>, value: KExpr<R>) =
    KArrayStore(array, index, value).intern()

fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>) =
    KArraySelect(array, index).intern()

fun <D : KSort, R : KSort> mkArrayConst(decl: KConstDecl<KArraySort<D, R>>) =
    KArrayConst(decl).intern()

fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
    mkArrayStore(this, index, value)

fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) =
    mkArraySelect(this, index)
