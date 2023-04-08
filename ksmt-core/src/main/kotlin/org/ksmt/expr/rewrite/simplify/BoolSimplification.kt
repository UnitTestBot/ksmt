package org.ksmt.expr.rewrite.simplify

import org.ksmt.KContext
import org.ksmt.expr.KAndExpr
import org.ksmt.expr.KApp
import org.ksmt.expr.KExpr
import org.ksmt.expr.KInterpretedValue
import org.ksmt.expr.KIteExpr
import org.ksmt.expr.KNotExpr
import org.ksmt.expr.KOrExpr
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KSort
import org.ksmt.utils.uncheckedCast

fun KContext.simplifyNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> = when (arg) {
    trueExpr -> falseExpr
    falseExpr -> trueExpr
    // (not (not x)) ==> x
    is KNotExpr -> arg.arg
    else -> mkNotNoSimplify(arg)
}

fun KContext.simplifyImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KExpr<KBoolSort> =
    simplifyOr(simplifyNot(p), q)

fun KContext.simplifyXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KExpr<KBoolSort> =
    simplifyEq(simplifyNot(a), b)

fun <T : KSort> KContext.simplifyEq(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    order: Boolean = true
): KExpr<KBoolSort> = when {
    lhs == rhs -> trueExpr
    lhs is KInterpretedValue<T> && rhs is KInterpretedValue<T> && lhs != rhs -> falseExpr
    lhs.sort == boolSort -> simplifyEqBool(lhs.uncheckedCast(), rhs.uncheckedCast(), order)
    order -> withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
    else -> mkEqNoSimplify(lhs, rhs)
}

fun <T : KSort> KContext.simplifyDistinct(
    args: List<KExpr<T>>,
    order: Boolean = true
): KExpr<KBoolSort> {
    if (args.isEmpty() || args.size == 1) return trueExpr

    // (distinct a b) ==> (not (= a b))
    if (args.size == 2) return simplifyNot(simplifyEq(args[0], args[1], order))

    // (distinct a b a) ==> false
    val distinctArgs = args.toSet()
    if (distinctArgs.size < args.size) return falseExpr

    // All arguments are not equal and all are interpreted values ==> all are distinct
    if (args.all { it is KInterpretedValue<*> }) return trueExpr

    return if (order) {
        val orderedArgs = args.toMutableList().apply {
            ensureExpressionsOrder()
        }
        mkDistinctNoSimplify(orderedArgs)
    } else {
        mkDistinctNoSimplify(args)
    }
}

fun <T : KSort> KContext.simplifyIte(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>
): KExpr<T> {
    var c = condition
    var t = trueBranch
    var e = falseBranch

    // (ite (not c) a b) ==> (ite c b a)
    if (c is KNotExpr) {
        c = c.arg
        val tmp = t
        t = e
        e = tmp
    }

    // (ite true t e) ==> t
    if (c == trueExpr) {
        return t
    }

    // (ite false t e) ==> e
    if (c == falseExpr) {
        return e
    }

    // (ite c (ite c t1 t2) t3)  ==> (ite c t1 t3)
    if (t is KIteExpr<T> && t.condition == c) {
        t = t.trueBranch
    }

    // (ite c t1 (ite c t2 t3))  ==> (ite c t1 t3)
    if (e is KIteExpr<T> && e.condition == c) {
        e = e.falseBranch
    }

    // (ite c t t) ==> t
    if (t == e) {
        return t
    }

    // (ite c t1 (ite c2 t1 t2)) ==> (ite (or c c2) t1 t2)
    if (e is KIteExpr<T> && e.trueBranch == t) {
        val simplifiedCondition = simplifyOr(c, e.condition)
        return simplifyIte(simplifiedCondition, t, e.falseBranch)
    }

    if (t.sort == boolSort) {
        return simplifyBoolIte(c, t.uncheckedCast(), e.uncheckedCast()).uncheckedCast()
    }

    return mkIteNoSimplify(c, t, e)
}


fun KContext.simplifyEqBool(
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    order: Boolean = true
): KExpr<KBoolSort> {
    // (= (not a) (not b)) ==> (= a b)
    if (lhs is KNotExpr && rhs is KNotExpr) {
        return simplifyEq(lhs.arg, rhs.arg, order)
    }

    when (lhs) {
        trueExpr -> return rhs
        falseExpr -> return simplifyNot(rhs)
    }

    when (rhs) {
        trueExpr -> return lhs
        falseExpr -> return simplifyNot(lhs)
    }

    // (= a (not a)) ==> false
    if (lhs.isComplement(rhs)) {
        return falseExpr
    }

    return if (order) {
        withExpressionsOrdered(lhs, rhs, ::mkEqNoSimplify)
    } else {
        mkEqNoSimplify(lhs, rhs)
    }
}

fun KExpr<KBoolSort>.isComplement(other: KExpr<KBoolSort>) =
    ctx.isComplementCore(this, other) || ctx.isComplementCore(other, this)

private fun KContext.isComplementCore(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>) =
    (a == trueExpr && b == falseExpr) || (a is KNotExpr && a.arg == b)

fun KContext.simplifyAnd(
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    flat: Boolean = true,
    order: Boolean = true
): KExpr<KBoolSort> = simplifyAndOr<KAndExpr>(
    flat = flat, order = order,
    lhs = lhs, rhs = rhs,
    // (and a b true) ==> (and a b)
    neutralElement = trueExpr,
    // (and a b false) ==> false
    zeroElement = falseExpr,
    buildResultBinaryExpr = { simplifiedLhs, simplifiedRhs -> mkAndNoSimplify(simplifiedLhs, simplifiedRhs) },
    buildResultFlatExpr = { simplifiedArgs -> mkAndNoSimplify(simplifiedArgs) }
)

fun KContext.simplifyAnd(
    args: List<KExpr<KBoolSort>>,
    flat: Boolean = true,
    order: Boolean = true
): KExpr<KBoolSort> = simplifyAndOr<KAndExpr>(
    flat = flat, order = order,
    args = args,
    // (and a b true) ==> (and a b)
    neutralElement = trueExpr,
    // (and a b false) ==> false
    zeroElement = falseExpr,
    buildResultBinaryExpr = { simplifiedLhs, simplifiedRhs -> mkAndNoSimplify(simplifiedLhs, simplifiedRhs) },
    buildResultFlatExpr = { simplifiedArgs -> mkAndNoSimplify(simplifiedArgs) }
)

fun KContext.simplifyOr(
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    flat: Boolean = true,
    order: Boolean = true
): KExpr<KBoolSort> = simplifyAndOr<KOrExpr>(
    flat = flat, order = order,
    lhs = lhs, rhs = rhs,
    // (or a b false) ==> (or a b)
    neutralElement = falseExpr,
    // (or a b true) ==> true
    zeroElement = trueExpr,
    buildResultBinaryExpr = { simplifiedLhs, simplifiedRhs -> mkOrNoSimplify(simplifiedLhs, simplifiedRhs) },
    buildResultFlatExpr = { simplifiedArgs -> mkOrNoSimplify(simplifiedArgs) }
)

fun KContext.simplifyOr(
    args: List<KExpr<KBoolSort>>,
    flat: Boolean = true,
    order: Boolean = true
): KExpr<KBoolSort> = simplifyAndOr<KOrExpr>(
    flat = flat, order = order,
    args = args,
    // (or a b false) ==> (or a b)
    neutralElement = falseExpr,
    // (or a b true) ==> true
    zeroElement = trueExpr,
    buildResultBinaryExpr = { simplifiedLhs, simplifiedRhs -> mkOrNoSimplify(simplifiedLhs, simplifiedRhs) },
    buildResultFlatExpr = { simplifiedArgs -> mkOrNoSimplify(simplifiedArgs) }
)

@Suppress("LongParameterList")
private inline fun <reified T : KApp<KBoolSort, KBoolSort>> simplifyAndOr(
    flat: Boolean,
    order: Boolean,
    args: List<KExpr<KBoolSort>>,
    neutralElement: KExpr<KBoolSort>,
    zeroElement: KExpr<KBoolSort>,
    buildResultBinaryExpr: (KExpr<KBoolSort>, KExpr<KBoolSort>) -> T,
    buildResultFlatExpr: (List<KExpr<KBoolSort>>) -> T
): KExpr<KBoolSort> = when(args.size){
    0 -> neutralElement
    1 -> args.single()
    2 -> simplifyAndOr(
        flat = flat, order = order,
        lhs = args.first(), rhs = args.last(),
        neutralElement = neutralElement,
        zeroElement = zeroElement,
        buildResultBinaryExpr = buildResultBinaryExpr,
        buildResultFlatExpr = buildResultFlatExpr
    )
    else -> simplifyAndOr(
        flat = flat, order = order,
        args = args,
        neutralElement = neutralElement,
        zeroElement = zeroElement,
        buildResultExpr = buildResultFlatExpr
    )
}

@Suppress("LongParameterList")
private inline fun <reified T : KApp<KBoolSort, KBoolSort>> simplifyAndOr(
    flat: Boolean,
    order: Boolean,
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    neutralElement: KExpr<KBoolSort>,
    zeroElement: KExpr<KBoolSort>,
    buildResultBinaryExpr: (KExpr<KBoolSort>, KExpr<KBoolSort>) -> T,
    buildResultFlatExpr: (List<KExpr<KBoolSort>>) -> T
): KExpr<KBoolSort> {
    if (flat && (lhs is T || rhs is T)) {
        return simplifyAndOr(
            flat = true, order = order,
            args = listOf(lhs, rhs),
            neutralElement = neutralElement,
            zeroElement = zeroElement
        ) { simplifiedArgs -> buildResultFlatExpr(simplifiedArgs) }
    }

    return when {
        lhs == rhs -> lhs
        lhs == zeroElement || rhs == zeroElement -> zeroElement
        lhs == neutralElement -> rhs
        rhs == neutralElement -> lhs
        lhs.isComplement(rhs) -> zeroElement
        order -> withExpressionsOrdered(lhs, rhs) { l, r -> buildResultBinaryExpr(l, r) }
        else -> buildResultBinaryExpr(lhs, rhs)
    }
}

@Suppress("NestedBlockDepth", "LongParameterList")
private inline fun <reified T : KApp<KBoolSort, KBoolSort>> simplifyAndOr(
    flat: Boolean,
    order: Boolean,
    args: List<KExpr<KBoolSort>>,
    neutralElement: KExpr<KBoolSort>,
    zeroElement: KExpr<KBoolSort>,
    buildResultExpr: (List<KExpr<KBoolSort>>) -> T
): KExpr<KBoolSort> {
    val posLiterals = HashSet<KExpr<KBoolSort>>(args.size)
    val negLiterals = HashSet<KExpr<KBoolSort>>(args.size)
    val resultArgs = ArrayList<KExpr<KBoolSort>>(args.size)

    for (arg in args) {
        if (flat && arg is T) {
            // flat expr one level
            for (flatArg in arg.args) {
                trySimplifyAndOrElement(flatArg, neutralElement, zeroElement, posLiterals, negLiterals, resultArgs)
                    ?.let { return it } // Expression was simplified
            }
        } else {
            trySimplifyAndOrElement(arg, neutralElement, zeroElement, posLiterals, negLiterals, resultArgs)
                ?.let { return it } // Expression was simplified
        }
    }

    return when (resultArgs.size) {
        0 -> neutralElement
        1 -> resultArgs.single()
        else -> {
            if (order) {
                resultArgs.ensureExpressionsOrder()
            }
            buildResultExpr(resultArgs)
        }
    }
}

@Suppress("LongParameterList")
private fun trySimplifyAndOrElement(
    arg: KExpr<KBoolSort>,
    neutralElement: KExpr<KBoolSort>,
    zeroElement: KExpr<KBoolSort>,
    posLiterals: MutableSet<KExpr<KBoolSort>>,
    negLiterals: MutableSet<KExpr<KBoolSort>>,
    resultElements: MutableList<KExpr<KBoolSort>>
): KExpr<KBoolSort>? {
    // (operation a b neutral) ==> (operation a b)
    if (arg == neutralElement) {
        return null
    }

    // (operation a b zero) ==> zero
    if (arg == zeroElement) {
        return zeroElement
    }

    if (arg is KNotExpr) {
        val lit = arg.arg
        // (operation (not a) b (not a)) ==> (operation (not a) b)
        if (!negLiterals.add(lit)) {
            return null
        }

        // (operation a (not a)) ==> zero
        if (lit in posLiterals) {
            return zeroElement
        }
    } else {
        // (operation a b a) ==> (operation a b)
        if (!posLiterals.add(arg)) {
            return null
        }

        // (operation a (not a)) ==> zero
        if (arg in negLiterals) {
            return zeroElement
        }
    }

    // element was not simplified
    resultElements += arg
    return null
}

private fun KContext.simplifyBoolIte(
    condition: KExpr<KBoolSort>,
    thenBranch: KExpr<KBoolSort>,
    elseBranch: KExpr<KBoolSort>
): KExpr<KBoolSort> {
    // (ite c true e) ==> (or c e)
    if (thenBranch == trueExpr) {
        if (elseBranch == falseExpr) {
            return condition
        }
        return simplifyOr(condition, elseBranch)
    }

    // (ite c false e) ==> (and (not c) e)
    if (thenBranch == falseExpr) {
        if (elseBranch == trueExpr) {
            return simplifyNot(condition)
        }
        return simplifyAnd(simplifyNot(condition), elseBranch)
    }

    // (ite c t true) ==> (or (not c) t)
    if (elseBranch == trueExpr) {
        return simplifyOr(simplifyNot(condition), thenBranch)
    }

    // (ite c t false) ==> (and c t)
    if (elseBranch == falseExpr) {
        return simplifyAnd(condition, thenBranch)
    }

    // (ite c t c) ==> (and c t)
    if (condition == elseBranch) {
        return simplifyAnd(condition, thenBranch)
    }

    // (ite c c e) ==> (or c e)
    if (condition == thenBranch) {
        return simplifyOr(condition, elseBranch)
    }

    return mkIteNoSimplify(condition, thenBranch, elseBranch)
}
