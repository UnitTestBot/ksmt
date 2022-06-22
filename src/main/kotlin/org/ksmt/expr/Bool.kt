package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort


class KAndExpr internal constructor(override val args: List<KExpr<KBoolSort>>) : KApp<KBoolSort, KExpr<KBoolSort>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkAndDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KOrExpr internal constructor(override val args: List<KExpr<KBoolSort>>) : KApp<KBoolSort, KExpr<KBoolSort>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkOrDecl()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KNotExpr internal constructor(val arg: KExpr<KBoolSort>) : KApp<KBoolSort, KExpr<KBoolSort>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkNotDecl()
    override val args: List<KExpr<KBoolSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KEqExpr<T : KSort> internal constructor(val lhs: KExpr<T>, val rhs: KExpr<T>) : KApp<KBoolSort, KExpr<T>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkEqDecl(lhs.sort)
    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

class KIteExpr<T : KSort> internal constructor(
    val condition: KExpr<KBoolSort>,
    val trueBranch: KExpr<T>,
    val falseBranch: KExpr<T>
) : KApp<T, KExpr<*>>() {
    override fun KContext.sort() = trueBranch.sort
    override fun KContext.decl() = mkIteDecl(trueBranch.sort)
    override val args: List<KExpr<*>>
        get() = listOf(condition, trueBranch, falseBranch)

    override fun accept(transformer: KTransformer): KExpr<T> = transformer.transform(this)
}

object KTrue : KApp<KBoolSort, KExpr<*>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkTrueDecl()
    override val args = emptyList<KExpr<*>>()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}

object KFalse : KApp<KBoolSort, KExpr<*>>() {
    override fun KContext.sort() = mkBoolSort()
    override fun KContext.decl() = mkFalseDecl()
    override val args = emptyList<KExpr<*>>()
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> = transformer.transform(this)
}
