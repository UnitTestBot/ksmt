package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KAddArithExpr
import io.ksmt.expr.KDivArithExpr
import io.ksmt.expr.KExpr
import io.ksmt.expr.KGeArithExpr
import io.ksmt.expr.KGtArithExpr
import io.ksmt.expr.KIntNumExpr
import io.ksmt.expr.KIsIntRealExpr
import io.ksmt.expr.KLeArithExpr
import io.ksmt.expr.KLtArithExpr
import io.ksmt.expr.KModIntExpr
import io.ksmt.expr.KMulArithExpr
import io.ksmt.expr.KPowerArithExpr
import io.ksmt.expr.KRealNumExpr
import io.ksmt.expr.KRemIntExpr
import io.ksmt.expr.KSubArithExpr
import io.ksmt.expr.KToIntRealExpr
import io.ksmt.expr.KToRealIntExpr
import io.ksmt.expr.KUnaryMinusArithExpr
import io.ksmt.sort.KArithSort
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KIntSort
import io.ksmt.sort.KRealSort
import io.ksmt.utils.ArithUtils.compareTo
import io.ksmt.utils.ArithUtils.toRealValue

interface KArithExprSimplifier : KExprSimplifierBase {

    fun simplifyEqInt(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KBoolSort> = with(ctx) {
        simplifyEqIntLight(lhs, rhs) { lhs2, rhs2 ->
            withExpressionsOrdered(lhs2, rhs2, ::mkEqNoSimplify)
        }
    }

    fun areDefinitelyDistinctInt(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): Boolean {
        if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
            return lhs.compareTo(rhs) != 0
        }
        return false
    }

    fun simplifyEqReal(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): KExpr<KBoolSort> = with(ctx) {
        simplifyEqRealLight(lhs, rhs) { lhs2, rhs2 ->
            withExpressionsOrdered(lhs2, rhs2, ::mkEqNoSimplify)
        }
    }

    fun areDefinitelyDistinctReal(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): Boolean {
        if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
            return lhs.toRealValue().compareTo(rhs.toRealValue()) != 0
        }
        return false
    }

    fun <T : KArithSort> KContext.preprocess(expr: KLtArithExpr<T>): KExpr<KBoolSort> = expr
    fun <T : KArithSort> KContext.postRewriteArithLt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyArithLt(lhs, rhs)

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteArithLt(lhs, rhs) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KLeArithExpr<T>): KExpr<KBoolSort> = expr
    fun <T : KArithSort> KContext.postRewriteArithLe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyArithLe(lhs, rhs)

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteArithLe(lhs, rhs) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KGtArithExpr<T>): KExpr<KBoolSort> = expr
    fun <T : KArithSort> KContext.postRewriteArithGt(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyArithGt(lhs, rhs)

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteArithGt(lhs, rhs) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KGeArithExpr<T>): KExpr<KBoolSort> = expr
    fun <T : KArithSort> KContext.postRewriteArithGe(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<KBoolSort> =
        simplifyArithGe(lhs, rhs)

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteArithGe(lhs, rhs) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KAddArithExpr<T>): KExpr<T> = expr
    fun <T : KArithSort> KContext.postRewriteArithAdd(args: List<KExpr<T>>): KExpr<T> =
        simplifyArithAdd(args)

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { args -> postRewriteArithAdd(args) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KMulArithExpr<T>): KExpr<T> = expr
    fun <T : KArithSort> KContext.postRewriteArithMul(args: List<KExpr<T>>): KExpr<T> =
        simplifyArithMul(args)

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { args -> postRewriteArithMul(args) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KSubArithExpr<T>): KExpr<T> {
        val args = expr.args

        return if (args.size == 1) {
            args.single()
        } else {
            val simplifiedArgs = arrayListOf(args.first())
            for (arg in args.drop(1)) {
                simplifiedArgs += KUnaryMinusArithExpr(this, arg)
            }
            KAddArithExpr(this, simplifiedArgs)
        }
    }

    fun <T : KArithSort> KContext.postRewriteArithSub(args: List<KExpr<T>>): KExpr<T> =
        error("Always preprocessed")

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            args = expr.args,
            preprocess = { preprocess(it) },
            simplifier = { args -> postRewriteArithSub(args) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KUnaryMinusArithExpr<T>): KExpr<T> = expr
    fun <T : KArithSort> KContext.postRewriteArithUnaryMinus(arg: KExpr<T>): KExpr<T> =
        simplifyArithUnaryMinus(arg)

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            preprocess = { preprocess(it) },
            simplifier = { arg -> postRewriteArithUnaryMinus(arg) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KDivArithExpr<T>): KExpr<T> = expr
    fun <T : KArithSort> KContext.postRewriteArithDiv(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyArithDiv(lhs, rhs)

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteArithDiv(lhs, rhs) }
        )

    fun <T : KArithSort> KContext.preprocess(expr: KPowerArithExpr<T>): KExpr<T> = expr
    fun <T : KArithSort> KContext.postRewriteArithPower(lhs: KExpr<T>, rhs: KExpr<T>): KExpr<T> =
        simplifyArithPower(lhs, rhs)

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteArithPower(lhs, rhs) }
        )

    fun KContext.preprocess(expr: KModIntExpr): KExpr<KIntSort> = expr
    fun KContext.postRewriteIntMod(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
        simplifyIntMod(lhs, rhs)

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteIntMod(lhs, rhs) }
        )

    fun KContext.preprocess(expr: KRemIntExpr): KExpr<KIntSort> = expr
    fun KContext.postRewriteIntRem(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): KExpr<KIntSort> =
        simplifyIntRem(lhs, rhs)

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.lhs,
            a1 = expr.rhs,
            preprocess = { preprocess(it) },
            simplifier = { lhs, rhs -> postRewriteIntRem(lhs, rhs) }
        )

    fun KContext.preprocess(expr: KToIntRealExpr): KExpr<KIntSort> = expr
    fun KContext.postRewriteRealToInt(arg: KExpr<KRealSort>): KExpr<KIntSort> =
        simplifyRealToInt(arg)

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            preprocess = { preprocess(it) },
            simplifier = { arg -> postRewriteRealToInt(arg) }
        )

    fun KContext.preprocess(expr: KIsIntRealExpr): KExpr<KBoolSort> = expr
    fun KContext.postRewriteRealIsInt(arg: KExpr<KRealSort>): KExpr<KBoolSort> =
        simplifyRealIsInt(arg)

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            preprocess = { preprocess(it) },
            simplifier = { arg -> postRewriteRealIsInt(arg) }
        )

    fun KContext.preprocess(expr: KToRealIntExpr): KExpr<KRealSort> = expr
    fun KContext.postRewriteIntToReal(arg: KExpr<KIntSort>): KExpr<KRealSort> =
        simplifyIntToReal(arg)

    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> =
        simplifyExpr(
            expr = expr,
            a0 = expr.arg,
            preprocess = { preprocess(it) },
            simplifier = { arg -> postRewriteIntToReal(arg) }
        )
}
