package io.ksmt.expr.rewrite.simplify

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.*
import io.ksmt.sort.*

fun isBoundTerm(assertion: KExpr<KBoolSort>): Boolean
{
    var curAssertion = assertion
    while (curAssertion is KNotExpr)
        curAssertion = curAssertion.arg
    return curAssertion is KQuantifier
}

fun qePreprocess(ctx: KContext,
                 assertions: List<KExpr<KBoolSort>>,
                 boundAssertions: ArrayList<KExistentialQuantifier>,
                 notAssertions: ArrayList<Boolean>,
                 freeAssertions: ArrayList<KExpr<KBoolSort>>)
{
    with (ctx)
    {
        for ((i, assertion) in assertions.withIndex()) {
            var notNumber = 0
            var newAssertion = assertion
            while (newAssertion is KNotExpr)
            {
                notNumber += 1
                newAssertion = newAssertion.arg
            }

            //For now, we assume that all bound variables are under the one quantifier
            if (newAssertion is KQuantifier)
            {
                val isUniversal = newAssertion is KUniversalQuantifier
                var body = newAssertion.body
                val bounds = newAssertion.bounds
                if (isUniversal)
                    body = mkNot(body)
                body = simplifyNot(body)
                newAssertion = mkExistentialQuantifier(body, bounds)
                if ((isUniversal and (notNumber % 2 == 0)) or
                    (notNumber % 2 == 1))
                    notAssertions[i] = true
                boundAssertions.add(newAssertion)
            }
            else
                freeAssertions.add(assertion)
        }
    }
}

fun qeProcess(ctx: KContext, assertion: KExistentialQuantifier): KExpr<KBoolSort>
{
    val qfAssertion = KTrue(ctx)
    var bounds = assertion.bounds
    val boundNum = bounds.size
    var curBound = bounds[boundNum - 1]
    var curAssert = assertion
    if (varOccursLinearly(curAssert, curBound))
        println()
    else
        println()
    return qfAssertion
}

fun varOccursLinearly(assertion: KExistentialQuantifier, bound: KDecl<*>): Boolean
{
    return true
}
