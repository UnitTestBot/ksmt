package org.ksmt.solver

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort

interface KModel {
    val declarations: Set<KDecl<*>>

    fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean = false): KExpr<T>

    fun <T : KSort> interpretation(decl: KDecl<T>): KFuncInterp<T>?

    fun detach(): KModel

    data class KFuncInterp<T : KSort>(
        val sort: T,
        val vars: List<KDecl<*>>,
        val entries: List<KFuncInterpEntry<T>>,
        val default: KExpr<T>?
    ){
        override fun toString(): String {
            if (entries.isEmpty()) return default.toString()
            return buildString {
                appendLine('{')
                entries.forEach { appendLine(it) }
                append("else -> ")
                appendLine(default)
                append('}')
            }
        }
    }

    data class KFuncInterpEntry<T : KSort>(
        val args: List<KExpr<*>>,
        val value: KExpr<T>
    ) {
        override fun toString(): String =
            args.joinToString(prefix = "(", postfix = ") -> $value")
    }
}
