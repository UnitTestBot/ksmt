package io.ksmt.solver.model

import io.ksmt.expr.KExpr
import io.ksmt.sort.KSort

sealed interface KFuncInterpEntry<T : KSort> {
    val args: List<KExpr<*>>
    val value: KExpr<T>
    val arity: Int

    companion object {
        fun printEntry(args: List<KExpr<*>>, value: KExpr<*>): String =
            args.joinToString(prefix = "(", postfix = ") -> $value")
    }
}

/**
 * Function interpretation entry, that may contain variables in
 * arguments or in a value.
 * */
sealed interface KFuncInterpEntryWithVars<T : KSort> : KFuncInterpEntry<T> {
    companion object {
        fun <T : KSort> create(
            args: List<KExpr<*>>,
            value: KExpr<T>
        ): KFuncInterpEntryWithVars<T> = when (args.size) {
            KFuncInterpEntryOneAry.ARITY -> KFuncInterpEntryWithVarsOneAry(args.single(), value)
            KFuncInterpEntryTwoAry.ARITY -> KFuncInterpEntryWithVarsTwoAry(
                args.first(),
                args.last(),
                value
            )

            KFuncInterpEntryThreeAry.ARITY -> {
                val (a0, a1, a2) = args
                KFuncInterpEntryWithVarsThreeAry(a0, a1, a2, value)
            }

            else -> KFuncInterpEntryWithVarsNAry(args, value)
        }
    }
}

/**
 * Function interpretation entry, that does NOT contain variables in
 * arguments and in a value.
 * For example, arguments and a value are interpreted values.
 * */
sealed interface KFuncInterpEntryVarsFree<T : KSort> : KFuncInterpEntry<T> {
    companion object {
        fun <T : KSort> create(
            args: List<KExpr<*>>,
            value: KExpr<T>
        ): KFuncInterpEntryVarsFree<T> = when (args.size) {
            KFuncInterpEntryOneAry.ARITY -> KFuncInterpEntryVarsFreeOneAry(args.single(), value)
            KFuncInterpEntryTwoAry.ARITY -> KFuncInterpEntryVarsFreeTwoAry(
                args.first(),
                args.last(),
                value
            )

            KFuncInterpEntryThreeAry.ARITY -> {
                val (a0, a1, a2) = args
                KFuncInterpEntryVarsFreeThreeAry(a0, a1, a2, value)
            }

            else -> KFuncInterpEntryVarsFreeNAry(args, value)
        }
    }
}

sealed interface KFuncInterpEntryOneAry<T : KSort> : KFuncInterpEntry<T> {
    val arg: KExpr<*>

    override val args: List<KExpr<*>>
        get() = listOf(arg)

    override val arity: Int
        get() = ARITY

    fun modify(arg: KExpr<*>, value: KExpr<T>): KFuncInterpEntryOneAry<T>

    companion object {
        const val ARITY = 1
    }
}

sealed interface KFuncInterpEntryTwoAry<T : KSort> : KFuncInterpEntry<T> {
    val arg0: KExpr<*>
    val arg1: KExpr<*>

    override val args: List<KExpr<*>>
        get() = listOf(arg0, arg1)

    override val arity: Int
        get() = ARITY

    fun modify(arg0: KExpr<*>, arg1: KExpr<*>, value: KExpr<T>): KFuncInterpEntryTwoAry<T>

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

    override val arity: Int
        get() = ARITY

    fun modify(arg0: KExpr<*>, arg1: KExpr<*>, arg2: KExpr<*>, value: KExpr<T>): KFuncInterpEntryThreeAry<T>

    companion object {
        const val ARITY = 3
    }
}

sealed interface KFuncInterpEntryNAry<T : KSort> : KFuncInterpEntry<T> {
    override val arity: Int
        get() = args.size

    fun modify(args: List<KExpr<*>>, value: KExpr<T>): KFuncInterpEntryNAry<T>
}

data class KFuncInterpEntryVarsFreeOneAry<T : KSort>(
    override val arg: KExpr<*>,
    override val value: KExpr<T>
) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryOneAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(arg: KExpr<*>, value: KExpr<T>): KFuncInterpEntryOneAry<T> =
        KFuncInterpEntryVarsFreeOneAry(arg, value)
}

data class KFuncInterpEntryVarsFreeTwoAry<T : KSort>(
    override val arg0: KExpr<*>,
    override val arg1: KExpr<*>,
    override val value: KExpr<T>
) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryTwoAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(arg0: KExpr<*>, arg1: KExpr<*>, value: KExpr<T>): KFuncInterpEntryTwoAry<T> =
        KFuncInterpEntryVarsFreeTwoAry(arg0, arg1, value)
}

data class KFuncInterpEntryVarsFreeThreeAry<T : KSort>(
    override val arg0: KExpr<*>,
    override val arg1: KExpr<*>,
    override val arg2: KExpr<*>,
    override val value: KExpr<T>
) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryThreeAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(arg0: KExpr<*>, arg1: KExpr<*>, arg2: KExpr<*>, value: KExpr<T>): KFuncInterpEntryThreeAry<T> =
        KFuncInterpEntryVarsFreeThreeAry(arg0, arg1, arg2, value)
}

data class KFuncInterpEntryVarsFreeNAry<T : KSort> internal constructor(
    override val args: List<KExpr<*>>,
    override val value: KExpr<T>
) : KFuncInterpEntryVarsFree<T>, KFuncInterpEntryNAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(args: List<KExpr<*>>, value: KExpr<T>): KFuncInterpEntryNAry<T> =
        KFuncInterpEntryVarsFreeNAry(args, value)
}

data class KFuncInterpEntryWithVarsOneAry<T : KSort>(
    override val arg: KExpr<*>,
    override val value: KExpr<T>
) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryOneAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(arg: KExpr<*>, value: KExpr<T>): KFuncInterpEntryOneAry<T> =
        KFuncInterpEntryWithVarsOneAry(arg, value)
}

data class KFuncInterpEntryWithVarsTwoAry<T : KSort>(
    override val arg0: KExpr<*>,
    override val arg1: KExpr<*>,
    override val value: KExpr<T>
) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryTwoAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(arg0: KExpr<*>, arg1: KExpr<*>, value: KExpr<T>): KFuncInterpEntryTwoAry<T> =
        KFuncInterpEntryWithVarsTwoAry(arg0, arg1, value)
}

data class KFuncInterpEntryWithVarsThreeAry<T : KSort>(
    override val arg0: KExpr<*>,
    override val arg1: KExpr<*>,
    override val arg2: KExpr<*>,
    override val value: KExpr<T>
) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryThreeAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(arg0: KExpr<*>, arg1: KExpr<*>, arg2: KExpr<*>, value: KExpr<T>): KFuncInterpEntryThreeAry<T> =
        KFuncInterpEntryWithVarsThreeAry(arg0, arg1, arg2, value)
}

data class KFuncInterpEntryWithVarsNAry<T : KSort> internal constructor(
    override val args: List<KExpr<*>>,
    override val value: KExpr<T>
) : KFuncInterpEntryWithVars<T>, KFuncInterpEntryNAry<T> {
    override fun toString(): String = KFuncInterpEntry.printEntry(args, value)
    override fun modify(args: List<KExpr<*>>, value: KExpr<T>): KFuncInterpEntryNAry<T> =
        KFuncInterpEntryWithVarsNAry(args, value)
}
