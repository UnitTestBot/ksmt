package org.ksmt.expr

import org.ksmt.decl.KArraySelectDecl
import org.ksmt.decl.KArrayStoreDecl
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStore<D : KSort, R : KSort> internal constructor(
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    val value: KExpr<R>
) : KArrayExpr<D, R>(KArrayStoreDecl(array.sort), listOf(array, index, value)) {
    override val sort = array.sort
    override fun accept(transformer: KTransformer): KExpr<KArraySort<D, R>> {
        TODO("Not yet implemented")
    }
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KApp<R, KExpr<*>>(KArraySelectDecl(array.sort.range), listOf(array, index)) {
    override val sort = array.sort.range
    override fun accept(transformer: KTransformer): KExpr<R> {
        TODO("Not yet implemented")
    }
}

fun <D : KSort, R : KSort> mkArrayStore(array: KExpr<KArraySort<D, R>>, index: KExpr<D>, value: KExpr<R>) =
    KArrayStore(array, index, value).intern()

fun <D : KSort, R : KSort> mkArraySelect(array: KExpr<KArraySort<D, R>>, index: KExpr<D>) =
    KArraySelect(array, index).intern()

fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.store(index: KExpr<D>, value: KExpr<R>) =
    mkArrayStore(this, index, value)

fun <D : KSort, R : KSort> KExpr<KArraySort<D, R>>.select(index: KExpr<D>) =
    mkArraySelect(this, index)
