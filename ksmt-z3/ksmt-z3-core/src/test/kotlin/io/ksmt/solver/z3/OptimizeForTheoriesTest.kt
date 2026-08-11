package io.ksmt.solver.z3

import io.ksmt.KContext
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.KTheory.UF
import io.ksmt.utils.mkConst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue

/**
 * ksmt keeps uninterpreted sort values distinct through values of a descriptor sort, so every query
 * constrains that sort whichever theories the caller declared. When the logic from
 * [io.ksmt.solver.KSolverConfiguration.optimizeForTheories] forbade it, Z3 either answered UNKNOWN
 * (`QF_UF`, `QF_LRA`, `QF_NRA`) or treated the descriptor sort as uninterpreted, which put it in
 * the model and made [KZ3Model.uninterpretedSorts] throw `ClassCastException` (every BV logic).
 *
 * 7 of the 26 emittable combinations were affected. {[UF], [Array], [BV]} is a complete declaration
 * of such a caller's own theories and still failed, since a caller cannot declare a theory ksmt adds
 * on its own behalf.
 */
class OptimizeForTheoriesTest {

    @Test
    fun checkAndModelWorkForEveryTheoryCombination() {
        val broken = ALL_THEORY_COMBINATIONS.mapNotNull { theories ->
            val outcome = runCatching { solveAndReadModel(theories) }
                .fold({ it }, { "threw ${it::class.simpleName}: ${it.message?.take(80)}" })
            outcome.takeIf { it != EXPECTED_OUTCOME }?.let { "${theories.describe()} -> $it" }
        }

        assertTrue(
            broken.isEmpty(),
            "Expected '$EXPECTED_OUTCOME' for every theory combination, but:\n" + broken.joinToString("\n")
        )
    }

    /**
     * The logic is fixed when the solver is created, and the first assertion already depends on it,
     * so it cannot be changed afterwards.
     */
    @Test
    fun optimizeForTheoriesIsRejectedAfterFirstAssert() {
        KContext().use { ctx ->
            KZ3Solver(ctx).use { solver ->
                solver.assert(ctx.boolSort.mkConst("a"))

                assertFailsWith<KSolverException> {
                    solver.configure { optimizeForTheories(setOf(BV), quantifiersAllowed = false) }
                }
            }
        }
    }

    /** Uninterpreted sort values must stay distinct whichever descriptor sort encodes them. */
    @Test
    fun uninterpretedSortValuesStayDistinctUnderBvDescriptor() {
        KContext().use { ctx ->
            KZ3Solver(ctx).use { solver ->
                solver.configure { optimizeForTheories(setOf(BV), quantifiersAllowed = false) }
                with(ctx) {
                    val ref = mkUninterpretedSort(UNINTERPRETED_SORT_NAME)
                    // two different values of the same sort cannot be equal
                    solver.assert(mkUninterpretedSortValue(ref, 0) eq mkUninterpretedSortValue(ref, 1))
                }
                assertEquals(KSolverStatus.UNSAT, solver.check())
            }
        }
    }

    private companion object {
        private const val UNINTERPRETED_SORT_NAME = "Ref"
        private const val VALUE_COUNT = 3
        private const val EXPECTED_OUTCOME = "SAT, model sorts=[$UNINTERPRETED_SORT_NAME]"

        private fun t(vararg theories: KTheory) = theories.toSet()

        /** Every combination [KZ3SolverLazyConfiguration] has a specialized Z3 solver for, plus none. */
        private val ALL_THEORY_COMBINATIONS: List<Set<KTheory>?> = listOf(
            null, emptySet(),
            t(Array), t(Array, BV), t(Array, LIA), t(Array, NIA), t(Array, UF, BV), t(Array, UF, LIA),
            t(Array, UF, LIA, LRA), t(Array, UF, NIA), t(Array, UF, NIA, NRA), t(BV), t(BV, FP), t(FP),
            t(FP, LRA), t(LIA), t(LIA, LRA), t(LRA), t(NIA), t(NIA, NRA), t(NRA), t(UF), t(UF, BV),
            t(UF, LIA), t(UF, LRA), t(UF, NIA), t(UF, NIA, NRA), t(UF, NRA),
        )

        private fun Set<KTheory>?.describe() =
            this?.map { it.name }?.sorted()?.joinToString(",")?.ifEmpty { "<empty>" } ?: "<null>"

        /**
         * The query is what a symbolic execution engine emits for a handful of concrete heap
         * references: distinct [io.ksmt.expr.KUninterpretedSortValue]s, each pinned to a constant.
         * The bitvector assertion makes [BV] an honest part of the declared theory sets.
         */
        private fun solveAndReadModel(theories: Set<KTheory>?): String =
            KContext().use { ctx ->
                KZ3Solver(ctx).use { solver ->
                    theories?.let { solver.configure { optimizeForTheories(it, quantifiersAllowed = false) } }

                    with(ctx) {
                        val ref = mkUninterpretedSort(UNINTERPRETED_SORT_NAME)
                        val consts = List(VALUE_COUNT) { i ->
                            ref.mkConst("c$i").also { solver.assert(it eq mkUninterpretedSortValue(ref, i)) }
                        }
                        for (i in consts.indices) {
                            for (j in i + 1 until consts.size) solver.assert(consts[i] neq consts[j])
                        }
                        solver.assert(bv32Sort.mkConst("x") eq mkBv(42))
                    }

                    val status = solver.check()
                    if (status != KSolverStatus.SAT) {
                        "$status"
                    } else {
                        "SAT, model sorts=${solver.model().uninterpretedSorts.map { it.name }}"
                    }
                }
            }
    }
}
