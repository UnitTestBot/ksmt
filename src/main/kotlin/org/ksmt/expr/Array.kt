package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KArraySort
import org.ksmt.sort.KSort

class KArrayStore<D : KSort, R : KSort> internal constructor(
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>,
    val value: KExpr<R>
) : KApp<KArraySort<D, R>, KExpr<*>>() {
    override fun KContext.sort() = array.sort
    override fun KContext.decl() = mkArrayStoreDecl(array.sort)
    override val args: List<KExpr<*>>
        get() = listOf(array, index, value)

    override fun accept(transformer: KTransformer): KExpr<KArraySort<D, R>> = transformer.transform(this)
}

class KArraySelect<D : KSort, R : KSort> internal constructor(
    val array: KExpr<KArraySort<D, R>>,
    val index: KExpr<D>
) : KApp<R, KExpr<*>>() {
    override fun KContext.sort() = array.sort.range
    override fun KContext.decl() = mkArraySelectDecl(array.sort)
    override val args: List<KExpr<*>>
        get() = listOf(array, index)

    override fun accept(transformer: KTransformer): KExpr<R> = transformer.transform(this)
}
