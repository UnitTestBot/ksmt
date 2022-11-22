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

    override fun transform(expr: KAndExpr): KExpr<KBoolSort> = simplifyApp(expr) { args ->
        with(ctx) {
            val posLiterals = hashSetOf<KExpr<*>>()
            val negLiterals = hashSetOf<KExpr<*>>()
            val argsToProcess = arrayListOf<KExpr<KBoolSort>>()
            argsToProcess += args
            val resultArgs = arrayListOf<KExpr<KBoolSort>>()
            while (argsToProcess.isNotEmpty()) {
                val arg = argsToProcess.removeLast()

                // (and a b true) ==> (and a b)
                if (arg == trueExpr) {
                    continue
                }

                // (and a b false) ==> false
                if (arg == falseExpr) {
                    return@simplifyApp falseExpr
                }

                // (and a (and b c)) ==> (and a b c)
                if (arg is KAndExpr) {
                    argsToProcess += arg.args
                    continue
                }

                if (arg is KNotExpr) {
                    val lit = arg.arg
                    // (and a b a) ==> (and a b)
                    if (!negLiterals.add(lit)) {
                        continue
                    }

                    // (and a (not a)) ==> false
                    if (lit in posLiterals) {
                        return@simplifyApp falseExpr
                    }
                } else {
                    // (and a b a) ==> (and a b)
                    if (!posLiterals.add(arg)) {
                        continue
                    }

                    // (and a (not a)) ==> false
                    if (arg in negLiterals) {
                        return@simplifyApp falseExpr
                    }
                }
                resultArgs += arg
            }
            when (resultArgs.size) {
                0 -> trueExpr
                1 -> resultArgs.single()
                else -> mkAnd(resultArgs)
            }
        }
    }

    override fun transform(expr: KOrExpr): KExpr<KBoolSort> = simplifyApp(expr) { args ->
        with(ctx) {
            val posLiterals = hashSetOf<KExpr<*>>()
            val negLiterals = hashSetOf<KExpr<*>>()
            val argsToProcess = arrayListOf<KExpr<KBoolSort>>()
            argsToProcess += args
            val resultArgs = arrayListOf<KExpr<KBoolSort>>()
            while (argsToProcess.isNotEmpty()) {
                val arg = argsToProcess.removeLast()

                // (or a b true) ==> true
                if (arg == trueExpr) {
                    return@simplifyApp trueExpr
                }

                // (or a b false) ==> (or a b)
                if (arg == falseExpr) {
                    continue
                }

                // (or a (or a c)) ==> (or a b c)
                if (arg is KOrExpr) {
                    argsToProcess += arg.args
                    continue
                }

                if (arg is KNotExpr) {
                    val lit = arg.arg
                    // (or a b a) ==> (or a b)
                    if (!negLiterals.add(lit)) {
                        continue
                    }

                    // (or a (not a)) ==> true
                    if (lit in posLiterals) {
                        return@simplifyApp trueExpr
                    }
                } else {
                    // (or a b a) ==> (or a b)
                    if (!posLiterals.add(arg)) {
                        continue
                    }

                    // (or a (not a)) ==> true
                    if (arg in negLiterals) {
                        return@simplifyApp trueExpr
                    }
                }
                resultArgs += arg
            }
            when (resultArgs.size) {
                0 -> falseExpr
                1 -> resultArgs.single()
                else -> mkOr(resultArgs)
            }
        }
    }

    override fun transform(expr: KNotExpr) = simplifyApp(expr) { (arg) ->
        with(ctx) {
            when (arg) {
                trueExpr -> falseExpr
                falseExpr -> trueExpr
                is KNotExpr -> arg.arg
                else -> mkNot(arg)
            }
        }
    }

    @Suppress("UNCHECKED_CAST")
    override fun <T : KSort> transform(expr: KIteExpr<T>): KExpr<T> =
        simplifyApp(expr as KApp<T, KExpr<KSort>>) { (condArg, thenArg, elseArg) ->
            with(ctx) {
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
                    )
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
                    // (ite c true e) ==> (or c e)
                    if (t == trueExpr) {
                        if (e == falseExpr) {
                            return@simplifyApp c.asExpr(expr.sort)
                        }
                        return@simplifyApp (c or e.asExpr(boolSort)).asExpr(expr.sort)
                    }

                    // (ite c false e) ==> (and (not c) e)
                    if (t == falseExpr) {
                        if (e == trueExpr) {
                            return@simplifyApp mkNot(c).asExpr(expr.sort)
                        }
                        return@simplifyApp (!c and e.asExpr(boolSort)).asExpr(expr.sort)
                    }

                    // (ite c t true) ==> (or (not c) t)
                    if (e == trueExpr) {
                        return@simplifyApp (!c or t.asExpr(boolSort)).asExpr(expr.sort)
                    }

                    // (ite c t false) ==> (and c t)
                    if (e == falseExpr) {
                        return@simplifyApp (c and t.asExpr(boolSort)).asExpr(expr.sort)
                    }

                    // (ite c t c) ==> (and c t)
                    if (c == e) {
                        return@simplifyApp (c and t.asExpr(boolSort)).asExpr(expr.sort)
                    }

                    // (ite c c e) ==> (or c e)
                    if (c == t) {
                        return@simplifyApp (c or e.asExpr(boolSort)).asExpr(expr.sort)
                    }
                }

                // todo: ite extra rules: bool_rewriter.cpp:846

                mkIte(c, t, e)
            }
        }

    override fun transform(expr: KImpliesExpr): KExpr<KBoolSort> = simplifyApp(expr) { (p, q) ->
        with(ctx) {
            mkNot(p) or q
        }
    }

    override fun transform(expr: KXorExpr): KExpr<KBoolSort> = simplifyApp(expr) { (a, b) ->
        with(ctx) {
            mkNot(a) eq b
        }
    }

    fun simplifyEqBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): KExpr<KBoolSort> = with(ctx) {
        if (lhs == rhs) return trueExpr

        var l = lhs
        var r = rhs

        // (= (not a) (not b)) ==> (= a b)
        if (l is KNotExpr && r is KNotExpr) {
            l = l.arg
            r = r.arg
        }

        when (l) {
            trueExpr -> return r
            falseExpr -> return !r
        }

        when (r) {
            trueExpr -> return l
            falseExpr -> return !l
        }

        // (= a (not a)) ==> false
        if (isComplement(l, r)) {
            return falseExpr
        }

        // todo: bool_rewriter.cpp:700

        mkEq(l, r)
    }

    fun areDefinitelyDistinctBool(lhs: KExpr<KBoolSort>, rhs: KExpr<KBoolSort>): Boolean =
        isComplement(lhs, rhs)

    private fun isComplement(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>) =
        isComplementCore(a, b) || isComplementCore(b, a)

    private fun isComplementCore(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>) = with(ctx) {
        (a == trueExpr && b == falseExpr) || (a is KNotExpr && a.arg == b)
    }
}
