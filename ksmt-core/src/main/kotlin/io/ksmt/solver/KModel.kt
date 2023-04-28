package io.ksmt.solver

import io.ksmt.decl.KDecl
import io.ksmt.expr.KExpr
import io.ksmt.expr.KUninterpretedSortValue
import io.ksmt.solver.model.KFuncInterp
import io.ksmt.sort.KSort
import io.ksmt.sort.KUninterpretedSort

interface KModel {
    val declarations: Set<KDecl<*>>

    val uninterpretedSorts: Set<KUninterpretedSort>

    fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean = false): KExpr<T>

    fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>?

    /**
     * Set of possible values of an Uninterpreted Sort.
     * */
    fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>?

    fun detach(): KModel
}
