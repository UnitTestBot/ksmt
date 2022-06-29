package org.ksmt.solver

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

interface KModel {
    val declarations: Set<KDecl<*>>
    fun <T : KSort> eval(expr: KExpr<T>, complete: Boolean = false): KExpr<T>
    fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>?
    fun detach(): KModel

    data class KFuncInterp<T : KSort>(
        val sort: T,
        val vars: List<KDecl<*>>,
        val entries: List<KFuncInterpEntry<T>>,
        val default: KExpr<T>?
    )

    data class KFuncInterpEntry<T : KSort>(
        val args: List<KExpr<*>>,
        val value: KExpr<T>
    )
}
