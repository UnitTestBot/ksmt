package org.ksmt.expr.rewrite.simplify

import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KImpliesExpr
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.expr.KXorExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.asExpr

interface KBoolExprSimplifier : KExprSimplifierBase {

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = simplifyApp(
        expr = expr,
        preprocess = {
            // (and a (and b c)) ==> (and a b c)
            val flatArgs = flatAnd(expr)
            if (flatArgs.size != expr.args.size) {
                mkAnd(flatArgs)
            } else {
                expr
            }
        }
    ) { args ->

        val resultArgs = simplifyAndOr(
            args = args,
            // (and a b true) ==> (and a b)
            neutralElement = trueExpr,
            // (and a b false) ==> false
            zeroElement = falseExpr,
            // (and a (not a)) ==> false
            complementValue = falseExpr
        )

        when (resultArgs.size) {
            0 -> trueExpr
            1 -> resultArgs.single()
            else -> mkAnd(resultArgs)
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = simplifyApp(
        expr = expr,
        preprocess = {
            // (or a (or b c)) ==> (or a b c)
            val flatArgs = flatOr(expr)
            if (flatArgs.size != expr.args.size) {
                mkOr(flatArgs)
            } else {
                expr
            }
        }
    ) { args ->

        val resultArgs = simplifyAndOr(
            args = args,
            // (or a b false) ==> (or a b)
            neutralElement = falseExpr,
            // (or a b true) ==> true
            zeroElement = trueExpr,
            // (or a (not a)) ==> true
            complementValue = trueExpr
        )

        when (resultArgs.size) {
            0 -> falseExpr
            1 -> resultArgs.single()
            else -> mkOr(resultArgs)
        }
    }

    @Suppress("LoopWithTooManyJumpStatements", "NestedBlockDepth")
    private fun simplifyAndOr(
        args: List<KExpr<KBoolSort>>,
        neutralElement: KExpr<KBoolSort>,
        zeroElement: KExpr<KBoolSort>,
        complementValue: KExpr<KBoolSort>
    ): List<KExpr<KBoolSort>> = with(ctx) {
        val posLiterals = hashSetOf<KExpr<*>>()
        val negLiterals = hashSetOf<KExpr<*>>()
        val resultArgs = arrayListOf<KExpr<KBoolSort>>()

        for (arg in args) {
            // (operation a b neutral) ==> (operation a b)
            if (arg == neutralElement) {
                continue
            }
            // (operation a b zero) ==> zero
            if (arg == zeroElement) {
                return listOf(zeroElement)
            }

            if (arg is KNotExpr) {
                val lit = arg.arg
                // (operation (not a) b (not a)) ==> (operation (not a) b)
                if (!negLiterals.add(lit)) {
                    continue
                }

                // (operation a (not a)) ==> complement
                if (lit in posLiterals) {
                    return listOf(complementValue)
                }
            } else {
                // (operation a b a) ==> (operation a b)
                if (!posLiterals.add(arg)) {
                    continue
                }

                // (operation a (not a)) ==> complement
                if (arg in negLiterals) {
                    return listOf(complementValue)
                }
            }
            resultArgs += arg
        }

        return resultArgs
    }

    override fun transform(expr: KNotExpr) = simplifyApp(expr) { (arg) ->
        when (arg) {
            trueExpr -> falseExpr
            falseExpr -> trueExpr
            // (not (not x)) ==> x
            is KNotExpr -> arg.arg
            else -> mkNot(arg)
        }
    }

    @Suppress("UNCHECKED_CAST", "ComplexMethod")
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KSort>>) { (condArg, thenArg, elseArg) ->
            var c = condArg.asExpr(boolSort)
            var t = thenArg.asExpr(expr.sort)
            var e = elseArg.asExpr(expr.sort)

            // (ite (not c) a b) ==> (ite c b a)
            if (c is KNotExpr) {
                c = c.arg
                val tmp = t
                t = e
                e = tmp
            }

            // (ite c (ite c t1 t2) t3)  ==> (ite c t1 t3)
            if (t is KIteExpr<*> && t.condition == c) {
                t = t.trueBranch.asExpr(t.sort)
            }

            // (ite c t1 (ite c t2 t3))  ==> (ite c t1 t3)
            if (e is KIteExpr<*> && e.condition == c) {
                e = e.falseBranch.asExpr(t.sort)
            }

            // (ite c t1 (ite c2 t1 t2)) ==> (ite (or c c2) t1 t2)
            if (e is KIteExpr<*> && e.trueBranch == t) {
                return@simplifyApp mkIte(
                    condition = c or e.condition,
                    trueBranch = t,
                    falseBranch = e.falseBranch.asExpr(expr.sort)
                ).also { rewrite(it) }
            }

            // (ite true t e) ==> t
            if (c == trueExpr) {
                return@simplifyApp t
            }

            // (ite false t e) ==> e
            if (c == falseExpr) {
                return@simplifyApp e
            }

            // (ite c t t) ==> t
            if (t == e) {
                return@simplifyApp t
            }

            if (t.sort == boolSort) {
                trySimplifyBoolIte(
                    condition = c,
                    thenBranch = t.asExpr(boolSort),
                    elseBranch = e.asExpr(boolSort)
                )?.let { simplified ->
                    return@simplifyApp simplified.asExpr(expr.sort).also { rewrite(it) }
                }
            }

            mkIte(c, t, e)
        }

    private fun trySimplifyBoolIte(
        condition: KExpr<KBoolSort>,
        thenBranch: KExpr<KBoolSort>,
        elseBranch: KExpr<KBoolSort>
    ): KExpr<KBoolSort>? = with(ctx) {
        // (ite c true e) ==> (or c e)
        if (thenBranch == trueExpr) {
            if (elseBranch == falseExpr) {
                return condition
            }
            return condition or elseBranch
        }

        // (ite c false e) ==> (and (not c) e)
        if (thenBranch == falseExpr) {
            if (elseBranch == trueExpr) {
                return !condition
            }
            return !condition and elseBranch
        }

        // (ite c t true) ==> (or (not c) t)
        if (elseBranch == trueExpr) {
            return !condition or thenBranch
        }

        // (ite c t false) ==> (and c t)
        if (elseBranch == falseExpr) {
            return condition and thenBranch
        }

        // (ite c t c) ==> (and c t)
        if (condition == elseBranch) {
            return condition and thenBranch
        }

        // (ite c c e) ==> (or c e)
        if (condition == thenBranch) {
            return condition or elseBranch
        }
        return null
    }

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = simplifyApp(
        expr = expr,
        preprocess = { !expr.p or expr.q }
    ) {
        error("Always preprocessed")
    }

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = simplifyApp(
        expr = expr,
        preprocess = { !expr.a eq expr.b }
    ) {
        error("Always preprocessed")
    }

    fun simplifyEqBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        // (= (not a) (not b)) ==> (= a b)
        if (lhs is KNotExpr && rhs is KNotExpr) {
            return mkEq(lhs.arg, rhs.arg).also { rewrite(it) }
        }

        when (lhs) {
            trueExpr -> return rhs
            falseExpr -> return !rhs.also { rewrite(it) }
        }

        when (rhs) {
            trueExpr -> return lhs
            falseExpr -> return !lhs.also { rewrite(it) }
        }

        // (= a (not a)) ==> false
        if (isComplement(lhs, rhs)) {
            return falseExpr
        }

        mkEq(lhs, rhs)
    }

    fun areDefinitelyDistinctBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): Boolean =
        isComplement(lhs, rhs)

    private fun isComplement(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>) =
        isComplementCore(a, b) || isComplementCore(b, a)

    private fun isComplementCore(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>) = with(ctx) {
        (a == trueExpr && b == falseExpr) || (a is KNotExpr && a.arg == b)
    }

    private fun flatAnd(expr: KAndExpr) = flatExpr<KAndExpr>(expr) { it.args }
    private fun flatOr(expr: KOrExpr) = flatExpr<KOrExpr>(expr) { it.args }

    private inline fun <reified T> flatExpr(
        initial: KExpr<KBoolSort>,
        getArgs: (T) -> List<KExpr<KBoolSort>>,
    ): List<KExpr<KBoolSort>> {
        val flatten = arrayListOf<KExpr<KBoolSort>>()
        val unprocessed = arrayListOf<KExpr<KBoolSort>>()
        unprocessed += initial
        while (unprocessed.isNotEmpty()) {
            val e = unprocessed.removeLast()
            if (e !is T) {
                flatten += e
                continue
            }
            unprocessed += getArgs(e).asReversed()
        }
        return flatten
    }
}
