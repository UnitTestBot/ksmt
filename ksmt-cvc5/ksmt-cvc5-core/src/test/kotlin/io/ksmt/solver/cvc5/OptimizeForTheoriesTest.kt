package io.ksmt.solver.cvc5

import io.ksmt.KContext
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
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
 * [io.ksmt.solver.KSolverConfiguration.optimizeForTheories] forbade it, cvc5 rejected the check --
 * `QF_UF`, `QF_UFBV`, `QF_AUFBV`, `QF_AX` and `QF_ABV` among them.
 *
 * Only theory sets containing [UF] or [Array] appear below, since those are the ones under which an
 * uninterpreted sort is legal at all. A regression here does not fail the test, it kills the JUnit
 * worker: on the Linux natives ksmt ships, cvc5's rejection cannot unwind out of libcvc5 and the
 * process dies at exit code 134 with no diagnostic.
 */
class OptimizeForTheoriesTest {

    @Test
    fun checkAndModelWorkForEveryTheoryCombination() {
        val broken = THEORY_COMBINATIONS_WITH_UNINTERPRETED_SORTS.mapNotNull { theories ->
            val outcome = runCatching { solveAndReadModel(theories) }
                .fold({ it }, { "threw ${it::class.simpleName}: ${it.message?.take(80)}" })
            outcome.takeIf { it != EXPECTED_OUTCOME }?.let { "${theories.describe()} -> $it" }
        }

        assertTrue(
            broken.isEmpty(),
            "Expected '$EXPECTED_OUTCOME' for every theory combination, but:\n" + broken.joinToString("\n")
        )
    }

    @Test
    fun optimizeForTheoriesIsRejectedAfterFirstAssert() {
        KContext().use { ctx ->
            KCvc5Solver(ctx).use { solver ->
                solver.assert(ctx.boolSort.mkConst("a"))

                assertFailsWith<KSolverException> {
                    solver.configure { optimizeForTheories(setOf(UF, BV), quantifiersAllowed = false) }
                }
            }
        }
    }

    /** Uninterpreted sort values must stay distinct whichever descriptor sort encodes them. */
    @Test
    fun uninterpretedSortValuesStayDistinctUnderBvDescriptor() {
        KContext().use { ctx ->
            KCvc5Solver(ctx).use { solver ->
                solver.configure { optimizeForTheories(setOf(UF, BV), quantifiersAllowed = false) }
                with(ctx) {
                    val ref = mkUninterpretedSort(UNINTERPRETED_SORT_NAME)
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

        private val THEORY_COMBINATIONS_WITH_UNINTERPRETED_SORTS: List<Set<KTheory>?> = listOf(
            null,
            t(Array), t(Array, BV), t(Array, LIA), t(Array, NIA), t(Array, UF), t(Array, UF, BV),
            t(Array, UF, LIA), t(Array, UF, LIA, LRA), t(Array, UF, NIA), t(Array, UF, NIA, NRA),
            t(UF), t(UF, BV), t(UF, LIA), t(UF, LRA), t(UF, NIA), t(UF, NIA, NRA), t(UF, NRA),
        )

        private fun Set<KTheory>?.describe() =
            this?.map { it.name }?.sorted()?.joinToString(",") ?: "<null>"

        private fun solveAndReadModel(theories: Set<KTheory>?): String =
            KContext().use { ctx ->
                KCvc5Solver(ctx).use { solver ->
                    theories?.let { solver.configure { optimizeForTheories(it, quantifiersAllowed = false) } }

                    with(ctx) {
                        val ref = mkUninterpretedSort(UNINTERPRETED_SORT_NAME)
                        val consts = List(VALUE_COUNT) { i ->
                            ref.mkConst("c$i").also { solver.assert(it eq mkUninterpretedSortValue(ref, i)) }
                        }
                        for (i in consts.indices) {
                            for (j in i + 1 until consts.size) solver.assert(consts[i] neq consts[j])
                        }
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
