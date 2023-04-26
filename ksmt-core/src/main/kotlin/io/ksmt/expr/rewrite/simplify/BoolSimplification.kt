package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.expr.KAndExpr
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.expr.KNotExpr
import io.ksmt.expr.KOrExpr
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort

fun KContext.simplifyNot(arg: KExpr<KBoolSort>): KExpr<KBoolSort> =
    simplifyNotLight(arg, ::mkNotNoSimplify)

fun KContext.simplifyImplies(p: KExpr<KBoolSort>, q: KExpr<KBoolSort>): KExpr<KBoolSort> =
    rewriteImplies(p, q, KContext::simplifyNot, KContext::simplifyOr)

fun KContext.simplifyXor(a: KExpr<KBoolSort>, b: KExpr<KBoolSort>): KExpr<KBoolSort> =
    rewriteXor(a, b, KContext::simplifyNot, KContext::simplifyEq)

fun <T : KSort> KContext.simplifyEq(
    lhs: KExpr<T>,
    rhs: KExpr<T>,
    order: Boolean = true
): KExpr<KBoolSort> =
    simplifyEqLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyEqBool(lhs2, rhs2, { l, r -> simplifyEqBool(l, r, order) }) { lhs3, rhs3 ->
            if (order) {
                withExpressionsOrdered(lhs3, rhs3, ::mkEqNoSimplify)
            } else {
                mkEqNoSimplify(lhs3, rhs3)
            }
        }
    }

fun <T : KSort> KContext.simplifyDistinct(
    args: List<KExpr<T>>,
    order: Boolean = true
): KExpr<KBoolSort> =
    simplifyDistinctLight(args, KContext::simplifyNot, { l, r -> simplifyEq(l, r, order) }) { args2 ->
        if (order) {
            val orderedArgs = args2.toMutableList().apply {
                ensureExpressionsOrder()
            }
            mkDistinctNoSimplify(orderedArgs)
        } else {
            mkDistinctNoSimplify(args2)
        }
    }

fun <T : KSort> KContext.simplifyIte(
    condition: KExpr<KBoolSort>,
    trueBranch: KExpr<T>,
    falseBranch: KExpr<T>
): KExpr<T> =
    simplifyIteNotCondition(condition, trueBranch, falseBranch) { condition2, trueBranch2, falseBranch2 ->
        simplifyIteLight(condition2, trueBranch2, falseBranch2) { condition3, trueBranch3, falseBranch3 ->
            simplifyIteSameBranches(
                condition3,
                trueBranch3,
                falseBranch3,
                KContext::simplifyIte,
                KContext::simplifyOr)
            { condition4, trueBranch4, falseBranch4 ->
                simplifyIteBool(condition4, trueBranch4, falseBranch4, KContext::simplifyBoolIte, ::mkIteNoSimplify)
            }
        }
    }


fun KContext.simplifyEqBool(
    lhs: KExpr<KBoolSort>,
    rhs: KExpr<KBoolSort>,
    order: Boolean = true
): KExpr<KBoolSort> =
    simplifyEqBoolLight(lhs, rhs) { lhs2, rhs2 ->
        simplifyEqBoolEvalConst(lhs2, rhs2) { lhs3, rhs3 ->
            simplifyEqBoolNot(lhs3, rhs3, KContext::simplifyEq) { lhs4, rhs4 ->
                if (order) {
                    withExpressionsOrdered(lhs4, rhs4, ::mkEqNoSimplify)
                } else {
                    mkEqNoSimplify(lhs4, rhs4)
                }
            }
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
    trueBranch: KExpr<KBoolSort>,
    falseBranch: KExpr<KBoolSort>
): KExpr<KBoolSort> =
    simplifyBoolIteConstBranches(
        condition = condition,
        trueBranch = trueBranch,
        falseBranch = falseBranch,
        rewriteOr = KContext::simplifyOr,
        rewriteAnd = KContext::simplifyAnd,
        rewriteNot = KContext::simplifyNot
    ) { condition2, trueBranch2, falseBranch2 ->
        simplifyBoolIteSameConditionBranch(
            condition = condition2,
            trueBranch = trueBranch2,
            falseBranch = falseBranch2,
            rewriteAnd = KContext::simplifyAnd,
            rewriteOr = KContext::simplifyOr,
            cont = ::mkIteNoSimplify
        )
    }
