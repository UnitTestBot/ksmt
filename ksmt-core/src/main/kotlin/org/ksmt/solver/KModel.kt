package org.ksmt.solver

import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.mkFreshConstDecl

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

    sealed interface KFuncInterpEntry<T : KSort> {
        val args: List<KExpr<*>>
        val value: KExpr<T>

        companion object {
            fun printEntry(args: List<KExpr<*>>, value: KExpr<*>): String =
                args.joinToString(prefix = "(", postfix = ") -> $value")
        }
    }

    sealed interface KFuncInterpEntryOneAry<T : KSort> : KFuncInterpEntry<T> {
        val arg: KExpr<*>

        override val args: List<KExpr<*>>
            get() = listOf(arg)

        companion object {
            const val ARITY = 1
        }
    }

    sealed interface KFuncInterpEntryTwoAry<T : KSort> : KFuncInterpEntry<T> {
        val arg0: KExpr<*>
        val arg1: KExpr<*>

        override val args: List<KExpr<*>>
            get() = listOf(arg0, arg1)

        companion object {
            const val ARITY = 2
        }
    }

    sealed interface KFuncInterpEntryThreeAry<T : KSort> : KFuncInterpEntry<T> {
        val arg0: KExpr<*>
        val arg1: KExpr<*>
        val arg2: KExpr<*>

        override val args: List<KExpr<*>>
            get() = listOf(arg0, arg1, arg2)

        companion object {
            const val ARITY = 3
        }
    }

    sealed interface KFuncInterpEntryNAry<T : KSort> : KFuncInterpEntry<T>

    sealed interface KFuncInterpEntryVarsFree<T : KSort> : KFuncInterpEntry<T>{
        companion object {
            fun <T : KSort> create(
                args: List<KExpr<*>>,
                value: KExpr<T>
            ): KFuncInterpEntryVarsFree<T> = when (args.size) {
                KFuncInterpEntryOneAry.ARITY -> KFuncInterpEntryVarsFreeOneAry(args.single(), value)
                KFuncInterpEntryTwoAry.ARITY -> KFuncInterpEntryVarsFreeTwoAry(args.first(), args.last(), value)
                KFuncInterpEntryThreeAry.ARITY -> {
                    val (a0, a1, a2) = args
                    KFuncInterpEntryVarsFreeThreeAry(a0, a1, a2, value)
                }

                else -> KFuncInterpEntryVarsFreeNAry(args, value)
            }
        }
    }

    sealed interface KFuncInterpEntryWithVars<T : KSort> : KFuncInterpEntry<T> {
        companion object {
            fun <T : KSort> create(
                args: List<KExpr<*>>,
                value: KExpr<T>
            ): KFuncInterpEntryWithVars<T> = when (args.size) {
                KFuncInterpEntryOneAry.ARITY -> KFuncInterpEntryWithVarsOneAry(args.single(), value)
                KFuncInterpEntryTwoAry.ARITY -> KFuncInterpEntryWithVarsTwoAry(args.first(), args.last(), value)
                KFuncInterpEntryThreeAry.ARITY -> {
                    val (a0, a1, a2) = args
                    KFuncInterpEntryWithVarsThreeAry(a0, a1, a2, value)
                }

                else -> KFuncInterpEntryWithVarsNAry(args, value)
            }
        }
    }

    data class KFuncInterpEntryVarsFreeOneAry<T : KSort>(
        override val arg: KExpr<*>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryOneAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryVarsFreeTwoAry<T : KSort>(
        override val arg0: KExpr<*>,
        override val arg1: KExpr<*>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryTwoAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryVarsFreeThreeAry<T : KSort>(
        override val arg0: KExpr<*>,
        override val arg1: KExpr<*>,
        override val arg2: KExpr<*>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryThreeAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryVarsFreeNAry<T : KSort> internal constructor(
        override val args: List<KExpr<*>>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryNAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryWithVarsOneAry<T : KSort>(
        override val arg: KExpr<*>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryOneAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryWithVarsTwoAry<T : KSort>(
        override val arg0: KExpr<*>,
        override val arg1: KExpr<*>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryTwoAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryWithVarsThreeAry<T : KSort>(
        override val arg0: KExpr<*>,
        override val arg1: KExpr<*>,
        override val arg2: KExpr<*>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryThreeAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }

    data class KFuncInterpEntryWithVarsNAry<T : KSort> internal constructor(
        override val args: List<KExpr<*>>,
        override val value: KExpr<T>
    ) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryNAry<T> {
        override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    }
}
