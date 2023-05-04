package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.expr.KInterpretedValue
import io.ksmt.expr.KIteExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import io.ksmt.utils.uncheckedCast

inline fun KContext.simplifyNotLight(
    arg: KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = when (arg) {
    trueExpr -> falseExpr
    falseExpr -> trueExpr
    // (not (not x)) ==> x
    is KNotExpr -> arg.arg
    else -> cont(arg)
}

/** (=> p q) ==> (or (not p) q) */
inline fun KContext.rewriteImplies(
    p: KExpr<KBoolSort>,
    q: KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    rewriteOr: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteOr(rewriteNot(p), q)

/** (xor a b) ==> (= (not a) b) */
inline fun KContext.rewriteXor(
    a: KExpr<KBoolSort>,
    b: KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    rewriteEq: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = rewriteEq(rewriteNot(a), b)

inline fun <T : KSort> KContext.simplifyEqLight(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = when {
    lhs == rhs -> trueExpr
    lhs is KInterpretedValue<T> && rhs is KInterpretedValue<T> && lhs != rhs -> falseExpr
    else -> cont(lhs, rhs)
}

inline fun <T : KSort> KContext.simplifyEqBool(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    rewriteEqBool: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> =
    if (lhs.sort == boolSort) {
        rewriteEqBool(lhs.uncheckedCast(), rhs.uncheckedCast())
    } else {
        cont(lhs, rhs)
    }

inline fun KContext.simplifyEqBoolLight(
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    // (= a (not a)) ==> false
    if (lhs.isComplement(rhs)) {
        return falseExpr
    }

    when (lhs) {
        trueExpr -> return rhs
        falseExpr -> return rewriteNot(rhs)
    }

    when (rhs) {
        trueExpr -> return lhs
        falseExpr -> return rewriteNot(lhs)
    }

    return cont(lhs, rhs)
}

/** (= (not a) (not b)) ==> (= a b) */
inline fun KContext.simplifyEqBoolNot(
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    rewriteEq: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (lhs is KNotExpr && rhs is KNotExpr) {
        return rewriteEq(lhs.arg, rhs.arg)
    }

    return cont(lhs, rhs)
}

inline fun <T : KSort> KContext.simplifyDistinctLight(
    args: List<KExpr<T>>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    rewriteEq: KContext.(KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    cont: (List<KExpr<T>>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> {
    if (args.isEmpty() || args.size == 1) return trueExpr

    // (distinct a b) ==> (not (= a b))
    if (args.size == 2) rewriteNot(rewriteEq(args[0], args[1]))

    // (distinct a b a) ==> false
    val distinctArgs = args.toSet()
    if (distinctArgs.size < args.size) return falseExpr

    // All arguments are not equal and all are interpreted values ==> all are distinct
    if (args.all { it is KInterpretedValue<*> }) return trueExpr

    return cont(args)
}

/**
 * (ite c true e) ==> (or c e)
 * (ite c false e) ==> (and (not c) e)
 * (ite c t true) ==> (or (not c) t)
 * (ite c t false) ==> (and c t)
 */
@Suppress("LongParameterList")
inline fun KContext.simplifyBoolIteConstBranches(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<KBoolSort>,
    falseBranch: KExpr<KBoolSort>,
    rewriteOr: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    rewriteAnd: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    rewriteNot: KContext.(KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>, KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = when {
    trueBranch == trueExpr && falseBranch == falseExpr -> condition
    trueBranch == trueExpr -> rewriteOr(condition, falseBranch)
    trueBranch == falseExpr && falseBranch == trueExpr -> rewriteNot(condition)
    trueBranch == falseExpr -> rewriteAnd(rewriteNot(condition), falseBranch)
    falseBranch == trueExpr -> rewriteOr(rewriteNot(condition), trueBranch)
    falseBranch == falseExpr -> rewriteAnd(condition, trueBranch)
    else -> cont(condition, trueBranch, falseBranch)
}

/**
 * (ite c t c) ==> (and c t)
 * (ite c c e) ==> (or c e)
 */
@Suppress("LongParameterList")
inline fun KContext.simplifyBoolIteSameConditionBranch(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<KBoolSort>,
    falseBranch: KExpr<KBoolSort>,
    rewriteAnd: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    rewriteOr: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>, KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>
): KExpr<KBoolSort> = when (condition) {
    falseBranch -> rewriteAnd(condition, trueBranch)
    trueBranch -> rewriteOr(condition, falseBranch)
    else -> cont(condition, trueBranch, falseBranch)
}

inline fun <T : KSort> KContext.simplifyIteLight(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>,
    cont: (KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> {
    var t = trueBranch
    var e = falseBranch

    // (ite true t e) ==> t
    if (condition == trueExpr) {
        return t
    }

    // (ite false t e) ==> e
    if (condition == falseExpr) {
        return e
    }

    // (ite c (ite c t1 t2) t3)  ==> (ite c t1 t3)
    if (t is KIteExpr<T> && t.condition == condition) {
        t = t.trueBranch
    }

    // (ite c t1 (ite c t2 t3))  ==> (ite c t1 t3)
    if (e is KIteExpr<T> && e.condition == condition) {
        e = e.falseBranch
    }

    // (ite c t t) ==> t
    if (t == e) {
        return t
    }

    return cont(condition, t, e)
}

/** (ite (not c) a b) ==> (ite c b a) */
inline fun <T : KSort> KContext.simplifyIteNotCondition(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>,
    cont: (KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> =
    if (condition is KNotExpr) {
        cont(condition.arg, falseBranch, trueBranch)
    } else {
        cont(condition, trueBranch, falseBranch)
    }

/** (ite c t1 (ite c2 t1 t2)) ==> (ite (or c c2) t1 t2) */
@Suppress("LongParameterList")
inline fun <T : KSort> KContext.simplifyIteSameBranches(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>,
    rewriteIte: KContext.(KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>,
    rewriteOr: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> =
    if (falseBranch is KIteExpr<T> && falseBranch.trueBranch == trueBranch) {
        rewriteIte(rewriteOr(condition, falseBranch.condition), trueBranch, falseBranch.falseBranch)
    } else {
        cont(condition, trueBranch, falseBranch)
    }

inline fun <T : KSort> KContext.simplifyIteBool(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>,
    rewriteBoolIte: KContext.(KExpr<KBoolSort>, KExpr<KBoolSort>, KExpr<KBoolSort>) -> KExpr<KBoolSort>,
    cont: (KExpr<KBoolSort>, KExpr<T>, KExpr<T>) -> KExpr<T>
): KExpr<T> =
    if (trueBranch.sort == boolSort) {
        rewriteBoolIte(condition, trueBranch.uncheckedCast(), falseBranch.uncheckedCast()).uncheckedCast()
    } else {
        cont(condition, trueBranch, falseBranch)
    }

@Suppress("LongParameterList")
inline fun <reified T : KApp<KBoolSort, KBoolSort>> simplifyAndOr(
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
inline fun <reified T : KApp<KBoolSort, KBoolSort>> simplifyAndOr(
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
inline fun <reified T : KApp<KBoolSort, KBoolSort>> simplifyAndOr(
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
fun trySimplifyAndOrElement(
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
