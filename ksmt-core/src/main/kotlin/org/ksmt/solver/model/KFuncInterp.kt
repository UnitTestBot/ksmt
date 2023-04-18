package org.ksmt.solver.model

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.sort.KSort
import org.ksmt.utils.mkFreshConstDecl

sealed interface KFuncInterp<T : KSort> {
    val decl: KDecl<T>
    val vars: List<KDecl<*>>
    val entries: List<KFuncInterpEntry<T>>
    val default: KExpr<T>?

    val sort: T
        get() = decl.sort

    companion object {
        fun checkVarsArity(vars: List<KDecl<*>>, arity: Int) {
            require(arity == vars.size) {
                "Function has $arity arguments but ${vars.size} were provided"
            }
        }

        fun checkEntriesArity(entries: List<KFuncInterpEntry<*>>, arity: Int) {
            require(entries.all { it.args.size == arity }) {
                "Function interpretation arguments mismatch"
            }
        }

        private const val ENTRY_SPACE_SHIFT = 4

        fun printEntries(entries: List<KFuncInterpEntry<*>>, default: KExpr<*>?): String {
            if (entries.isEmpty()) return default.toString()

            val spaces = " ".repeat(ENTRY_SPACE_SHIFT)
            return buildString {
                appendLine('{')
                entries.forEach {
                    append(spaces)
                    appendLine(it)
                }

                append(spaces)
                append("else -> ")
                appendLine(default)

                append('}')
            }
        }
    }
}

data class KFuncInterpWithVars<T : KSort>(
    override val decl: KDecl<T>,
    override val vars: List<KDecl<*>>,
    override val entries: List<KFuncInterpEntry<T>>,
    override val default: KExpr<T>?
) : KFuncInterp<T> {
    init {
        KFuncInterp.checkVarsArity(vars, decl.argSorts.size)
        KFuncInterp.checkEntriesArity(entries, decl.argSorts.size)
    }

    override fun toString(): String = KFuncInterp.printEntries(entries, default)
}

data class KFuncInterpVarsFree<T : KSort>(
    override val decl: KDecl<T>,
    override val entries: List<KFuncInterpEntryVarsFree<T>>,
    override val default: KExpr<T>?
) : KFuncInterp<T> {
    init {
        KFuncInterp.checkEntriesArity(entries, decl.argSorts.size)
    }

    override val vars: List<KDecl<*>> by lazy {
        decl.argSorts.map { it.mkFreshConstDecl("x") }
    }

    override fun toString(): String = KFuncInterp.printEntries(entries, default)
}
