package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KAddArithExpr<T : KArithSort<T>> internal constructor(
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>() {
    init {
        require(args.isNotEmpty())
    }

    override fun KContext.sort(): T = args.first().sort
    override fun KContext.decl() = mkArithAddDecl(sort)
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KMulArithExpr<T : KArithSort<T>> internal constructor(
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>() {
    init {
        require(args.isNotEmpty())
    }

    override fun KContext.sort(): T = args.first().sort
    override fun KContext.decl() = mkArithMulDecl(sort)
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KSubArithExpr<T : KArithSort<T>> internal constructor(
    override val args: List<KExpr<T>>
) : KApp<T, KExpr<T>>() {
    init {
        require(args.isNotEmpty())
    }

    override fun KContext.sort(): T = args.first().sort
    override fun KContext.decl() = mkArithSubDecl(sort)
    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KUnaryMinusArithExpr<T : KArithSort<T>> internal constructor(
    val arg: KExpr<T>
) : KApp<T, KExpr<T>>() {
    override fun KContext.sort(): T = arg.sort
    override fun KContext.decl() = mkArithUnaryMinusDecl(sort)
    override val args: List<KExpr<T>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KDivArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, KExpr<T>>() {
    override fun KContext.sort(): T = lhs.sort
    override fun KContext.decl() = mkArithDivDecl(sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KPowerArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<T, KExpr<T>>() {
    override fun KContext.sort(): T = lhs.sort
    override fun KContext.decl() = mkArithPowerDecl(sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

class KLtArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkArithLtDecl(lhs.sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KLeArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkArithLeDecl(lhs.sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KGtArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkArithGtDecl(lhs.sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KGeArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkArithGeDecl(lhs.sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
