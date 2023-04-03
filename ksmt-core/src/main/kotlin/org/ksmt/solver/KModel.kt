package org.ksmt.solver

import org.ksmt.decl.KDecl
import org.ksmt.decl.KFuncDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort

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

    data class KFuncInterp<T : KSort>(
        val decl: KDecl<T>,
        val vars: List<KDecl<*>>,
        val entries: List<KFuncInterpEntry<T>>,
        val default: KExpr<T>?
    ) {
        init {
            if (decl is KFuncDecl<T>) {
                require(decl.argSorts.size == vars.size) {
                    "Function $decl has ${decl.argSorts.size} arguments but ${vars.size} were provided"
                }
            }
            require(entries.all { it.args.size == vars.size }) {
                "Function interpretation arguments mismatch"
            }
        }

        val sort: T
            get() = decl.sort

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
