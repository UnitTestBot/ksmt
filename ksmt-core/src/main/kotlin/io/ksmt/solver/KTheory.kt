package io.ksmt.solver

import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.KTheory.UF
import io.ksmt.solver.KTheory.S

/**
 * SMT theory
 * */
enum class KTheory {
    UF, BV, FP, Array,
    LIA, NIA, LRA, NRA,
    S
}

@Suppress("ComplexMethod", "ComplexCondition")
fun Set<KTheory>?.smtLib2String(quantifiersAllowed: Boolean = false): String = buildString {
    val theories = this@smtLib2String

    if (!quantifiersAllowed) {
        append("QF_")
    }

    if (theories == null) {
        append("ALL")
        return@buildString
    }

    if (theories.isEmpty()) {
        append("SAT")
        return@buildString
    }

    if (Array in theories) {
        if (theories.size == 1) {
            append("AX")
            return@buildString
        }
        append("A")
    }

    if (UF in theories) {
        append("UF")
    }

    if (BV in theories) {
        append("BV")
    }

    if (FP in theories) {
        append("FP")
    }

    if (S in theories) {
        append("S")
    }

    if (LIA in theories || NIA in theories || LRA in theories || NRA in theories) {
        val hasNonLinear = NIA in theories || NRA in theories
        val hasReal = LRA in theories || NRA in theories
        val hasInt = LIA in theories || NIA in theories

        if (hasNonLinear) {
            append("N")
        } else {
            append("L")
        }

        if (hasInt) {
            append("I")
        }

        if (hasReal) {
            append("R")
        }

        append("A")
    }

}
