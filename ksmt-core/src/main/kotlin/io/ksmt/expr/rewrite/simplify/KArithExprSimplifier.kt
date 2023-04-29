package io.ksmt.expr.rewrite.simplify

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
        if (lhs == rhs) return trueExpr

        if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
            return (lhs.compareTo(rhs) == 0).expr
        }

        withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
    }

    fun areDefinitelyDistinctInt(lhs: KExpr<KIntSort>, rhs: KExpr<KIntSort>): Boolean {
        if (lhs is KIntNumExpr && rhs is KIntNumExpr) {
            return lhs.compareTo(rhs) != 0
        }
        return false
    }

    fun simplifyEqReal(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
            return (lhs.toRealValue().compareTo(rhs.toRealValue()) == 0).expr
        }

        return withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
    }

    fun areDefinitelyDistinctReal(lhs: KExpr<KRealSort>, rhs: KExpr<KRealSort>): Boolean {
        if (lhs is KRealNumExpr && rhs is KRealNumExpr) {
            return lhs.toRealValue().compareTo(rhs.toRealValue()) != 0
        }
        return false
    }

    override fun <T : KArithSort> transform(expr: KLtArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
            simplifyArithLt(lhs, rhs)
        }

    override fun <T : KArithSort> transform(expr: KLeArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
            simplifyArithLe(lhs, rhs)
        }

    override fun <T : KArithSort> transform(expr: KGtArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
            simplifyArithGt(lhs, rhs)
        }

    override fun <T : KArithSort> transform(expr: KGeArithExpr<T>): KExpr<KBoolSort> =
        simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
            simplifyArithGe(lhs, rhs)
        }

    override fun <T : KArithSort> transform(expr: KAddArithExpr<T>): KExpr<T> = simplifyExpr(expr, expr.args) { args ->
        simplifyArithAdd(args)
    }

    override fun <T : KArithSort> transform(expr: KMulArithExpr<T>): KExpr<T> = simplifyExpr(expr, expr.args) { args ->
        simplifyArithMul(args)
    }

    override fun <T : KArithSort> transform(expr: KSubArithExpr<T>): KExpr<T> =
        simplifyExpr(
            expr = expr,
            preprocess = {
                val args = expr.args
                if (args.size == 1) {
                    args.single()
                } else {
                    val simplifiedArgs = arrayListOf(args.first())
                    for (arg in args.drop(1)) {
                        simplifiedArgs += KUnaryMinusArithExpr(this, arg)
                    }
                    KAddArithExpr(this, simplifiedArgs)
                }
            }
        )

    override fun <T : KArithSort> transform(expr: KUnaryMinusArithExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.arg) { arg ->
            simplifyArithUnaryMinus(arg)
        }

    override fun <T : KArithSort> transform(expr: KDivArithExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
            simplifyArithDiv(lhs, rhs)
        }

    override fun <T : KArithSort> transform(expr: KPowerArithExpr<T>): KExpr<T> =
        simplifyExpr(expr, expr.lhs, expr.rhs) { base, power ->
            simplifyArithPower(base, power)
        }

    override fun transform(expr: KModIntExpr): KExpr<KIntSort> = simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
        simplifyIntMod(lhs, rhs)
    }

    override fun transform(expr: KRemIntExpr): KExpr<KIntSort> = simplifyExpr(expr, expr.lhs, expr.rhs) { lhs, rhs ->
        simplifyIntRem(lhs, rhs)
    }

    override fun transform(expr: KToIntRealExpr): KExpr<KIntSort> = simplifyExpr(expr, expr.arg) { arg ->
        simplifyRealToInt(arg)
    }

    override fun transform(expr: KIsIntRealExpr): KExpr<KBoolSort> = simplifyExpr(expr, expr.arg) { arg ->
        simplifyRealIsInt(arg)
    }

    override fun transform(expr: KToRealIntExpr): KExpr<KRealSort> = simplifyExpr(expr, expr.arg) { arg ->
        simplifyIntToReal(arg)
    }
}
