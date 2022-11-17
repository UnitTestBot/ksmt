package org.ksmt.expr

import org.ksmt.KContext
import org.ksmt.decl.KAndDecl
import org.ksmt.decl.KDistinctDecl
import org.ksmt.decl.KEqDecl
import org.ksmt.decl.KFalseDecl
import org.ksmt.decl.KImpliesDecl
import org.ksmt.decl.KIteDecl
import org.ksmt.decl.KNotDecl
import org.ksmt.decl.KOrDecl
import org.ksmt.decl.KTrueDecl
import org.ksmt.decl.KXorDecl
import org.ksmt.expr.transformer.KTransformerBase
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort

class KAndExpr internal constructor(
    ctx: KContext,
    override val args: List<KExpr<KBoolSort>>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KAndDecl = ctx.mkAndDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KOrExpr internal constructor(
    ctx: KContext,
    override val args: List<KExpr<KBoolSort>>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KOrDecl = ctx.mkOrDecl()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KNotExpr internal constructor(
    ctx: KContext,
    val arg: KExpr<KBoolSort>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KNotDecl = ctx.mkNotDecl()

    override val args: List<KExpr<KBoolSort>>
        get() = listOf(arg)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KImpliesExpr internal constructor(
    ctx: KContext,
    val p: KExpr<KBoolSort>,
    val q: KExpr<KBoolSort>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KImpliesDecl = ctx.mkImpliesDecl()

    override val args: List<KExpr<KBoolSort>>
        get() = listOf(p, q)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KXorExpr internal constructor(
    ctx: KContext,
    val a: KExpr<KBoolSort>,
    val b: KExpr<KBoolSort>
) : KApp<KBoolSort, KExpr<KBoolSort>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KXorDecl = ctx.mkXorDecl()

    override val args: List<KExpr<KBoolSort>>
        get() = listOf(a, b)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KEqExpr<T : KSort> internal constructor(
    ctx: KContext,
    val lhs: KExpr<T>, val rhs: KExpr<T>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KEqDecl<T> = with(ctx) { mkEqDecl(lhs.sort) }

    override val args: List<KExpr<T>>
        get() = listOf(lhs, rhs)

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KDistinctExpr<T : KSort> internal constructor(
    ctx: KContext,
    override val args: List<KExpr<T>>
) : KApp<KBoolSort, KExpr<T>>(ctx) {
    init {
        require(args.isNotEmpty()) { "distinct requires at least a single argument" }
    }

    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KDistinctDecl<T> = with(ctx) { mkDistinctDecl(args.first().sort) }

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KIteExpr<T : KSort> internal constructor(
    ctx: KContext,
    val condition: KExpr<KBoolSort>,
    val trueBranch: KExpr<T>,
    val falseBranch: KExpr<T>
) : KApp<T, KExpr<*>>(ctx) {

    override fun decl(): KIteDecl<T> = with(ctx) { mkIteDecl(trueBranch.sort) }

    override val args: List<KExpr<*>>
        get() = listOf(condition, trueBranch, falseBranch)

    override fun accept(transformer: KTransformerBase): KExpr<T> = transformer.transform(this)

    override val sort: T
        get() = ctx.getExprSort(this)

    override fun computeExprSort(): T = trueBranch.sort

    override fun sortComputationExprDependency(dependency: MutableList<KExpr<*>>) {
        dependency += trueBranch
    }
}

class KTrue(ctx: KContext) : KApp<KBoolSort, KExpr<*>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KTrueDecl = ctx.mkTrueDecl()

    override val args = emptyList<KExpr<*>>()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}

class KFalse(ctx: KContext) : KApp<KBoolSort, KExpr<*>>(ctx) {
    override val sort: KBoolSort
        get() = ctx.boolSort

    override fun decl(): KFalseDecl = ctx.mkFalseDecl()

    override val args = emptyList<KExpr<*>>()

    override fun accept(transformer: KTransformerBase): KExpr<KBoolSort> = transformer.transform(this)
}
