package org.ksmt.expr

import org.ksmt.decl.*
import org.ksmt.expr.manager.ExprManager.intern
import org.ksmt.expr.transformer.KArithTransformer
import org.ksmt.expr.transformer.KTransformer
import org.ksmt.sort.KArithSort
import org.ksmt.sort.KBoolSort

class KAddArithExpr<T : KArithSort<T>> internal constructor(
    args: List<KExpr<T>>
) : KArithExpr<T, KExpr<T>>(args) {
    init {
        require(args.isNotEmpty())
    }

    override val sort by lazy { args.first().sort }
    override val decl by lazy { KArithAddDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> {
        transformer as KArithTransformer
        val transformedArgs = args.map { it.accept(transformer) }
        if (transformedArgs == args) return transformer.transformArithAdd(this)
        return transformer.transformArithAdd(mkArithAdd(transformedArgs))
    }
}

class KMulArithExpr<T : KArithSort<T>> internal constructor(
    args: List<KExpr<T>>
) : KArithExpr<T, KExpr<T>>(args) {
    init {
        require(args.isNotEmpty())
    }

    override val sort by lazy { args.first().sort }
    override val decl by lazy { KArithMulDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

class KSubArithExpr<T : KArithSort<T>> internal constructor(
    args: List<KExpr<T>>
) : KArithExpr<T, KExpr<T>>(args) {
    init {
        require(args.isNotEmpty())
    }

    override val sort by lazy { args.first().sort }
    override val decl by lazy { KArithSubDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

class KUnaryMinusArithExpr<T : KArithSort<T>> internal constructor(
    val arg: KExpr<T>
) : KArithExpr<T, KExpr<T>>(listOf(arg)) {
    override val sort by lazy { arg.sort }
    override val decl by lazy { KArithUnaryMinusDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

class KDivArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KArithExpr<T, KExpr<T>>(listOf(lhs, rhs)) {
    override val sort by lazy { lhs.sort }
    override val decl by lazy { KArithDivDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

class KPowerArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KArithExpr<T, KExpr<T>>(listOf(lhs, rhs)) {
    override val sort by lazy { lhs.sort }
    override val decl by lazy { KArithPowerDecl(sort) }
    override fun accept(transformer: KTransformer): KExpr<T> {
        TODO("Not yet implemented")
    }
}

class KLtArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(listOf(lhs, rhs)) {
    override val decl by lazy { KArithLtDecl(lhs.sort) }
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

class KLeArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(listOf(lhs, rhs)) {
    override val decl by lazy { KArithLeDecl(lhs.sort) }
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

class KGtArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(listOf(lhs, rhs)) {
    override val decl by lazy { KArithGtDecl(lhs.sort) }
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

class KGeArithExpr<T : KArithSort<T>> internal constructor(
    val lhs: KExpr<T>,
    val rhs: KExpr<T>
) : KBoolExpr<KExpr<T>>(listOf(lhs, rhs)) {
    override val decl by lazy { KArithGeDecl(lhs.sort) }
    override fun accept(transformer: KTransformer): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }
}

fun <T : KArithSort<T>> mkArithAdd(vararg args: KExpr<T>) = KAddArithExpr(args.toList()).intern()
fun <T : KArithSort<T>> mkArithAdd(args: List<KExpr<T>>) = KAddArithExpr(args).intern()
fun <T : KArithSort<T>> mkArithMul(vararg args: KExpr<T>) = KMulArithExpr(args.toList()).intern()
fun <T : KArithSort<T>> mkArithMul(args: List<KExpr<T>>) = KMulArithExpr(args).intern()
fun <T : KArithSort<T>> mkArithSub(vararg args: KExpr<T>) = KSubArithExpr(args.toList()).intern()
fun <T : KArithSort<T>> mkArithSub(args: List<KExpr<T>>) = KSubArithExpr(args).intern()

fun <T : KArithSort<T>> mkArithUnaryMinus(arg: KExpr<T>) = KUnaryMinusArithExpr(arg).intern()
fun <T : KArithSort<T>> mkArithDiv(lhs: KExpr<T>, rhs: KExpr<T>) = KDivArithExpr(lhs, rhs).intern()
fun <T : KArithSort<T>> mkArithPower(lhs: KExpr<T>, rhs: KExpr<T>) = KPowerArithExpr(lhs, rhs).intern()

fun <T : KArithSort<T>> mkArithLt(lhs: KExpr<T>, rhs: KExpr<T>) = KLtArithExpr(lhs, rhs).intern()
fun <T : KArithSort<T>> mkArithLe(lhs: KExpr<T>, rhs: KExpr<T>) = KLeArithExpr(lhs, rhs).intern()
fun <T : KArithSort<T>> mkArithGt(lhs: KExpr<T>, rhs: KExpr<T>) = KGtArithExpr(lhs, rhs).intern()
fun <T : KArithSort<T>> mkArithGe(lhs: KExpr<T>, rhs: KExpr<T>) = KGeArithExpr(lhs, rhs).intern()

operator fun <T : KArithSort<T>> KExpr<T>.plus(other: KExpr<T>) = mkArithAdd(this, other)
operator fun <T : KArithSort<T>> KExpr<T>.times(other: KExpr<T>) = mkArithMul(this, other)
operator fun <T : KArithSort<T>> KExpr<T>.minus(other: KExpr<T>) = mkArithSub(this, other)
operator fun <T : KArithSort<T>> KExpr<T>.unaryMinus() = mkArithUnaryMinus(this)
operator fun <T : KArithSort<T>> KExpr<T>.div(other: KExpr<T>) = mkArithDiv(this, other)
fun <T : KArithSort<T>> KExpr<T>.power(other: KExpr<T>) = mkArithPower(this, other)

infix fun <T : KArithSort<T>> KExpr<T>.lt(other: KExpr<T>) = mkArithLt(this, other)
infix fun <T : KArithSort<T>> KExpr<T>.le(other: KExpr<T>) = mkArithLe(this, other)
infix fun <T : KArithSort<T>> KExpr<T>.gt(other: KExpr<T>) = mkArithGt(this, other)
infix fun <T : KArithSort<T>> KExpr<T>.ge(other: KExpr<T>) = mkArithGe(this, other)
