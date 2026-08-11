package io.ksmt.solver.cvc5

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.KTheory
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBv64Sort
import io.ksmt.sort.KBv8Sort
import io.ksmt.utils.mkConst
import kotlin.test.Test
import kotlin.test.assertEquals

private typealias ByteArraySort = KArraySort<KBv64Sort, KBv8Sort>

/**
 * Guards the `arrays-exp` option [KCvc5Solver] sets. [KContext.mkArrayConst] is internalized as
 * cvc5's `STORE_ALL`, which cvc5 refuses without that option:
 * *"Cannot handle assertion with term of kind STORE_ALL in this configuration. Try --arrays-exp."*
 *
 * Only a store over a const array read at a symbolic index reaches the refusal -- with no store, or
 * at a concrete index, cvc5's rewriter eliminates the const array first, and over an uninterpreted
 * base there is no `STORE_ALL` at all. The other tests pin that, so a regression narrows to either
 * the option or the rewriter.
 *
 * There is no test for the unset option: on the Linux natives ksmt ships, each library statically
 * links its own libstdc++, so the refusal cannot unwind out of libcvc5 and the process dies at exit
 * code 134 with no diagnostic, taking the JUnit worker with it.
 */
class ConstArrayTest {

    @Test
    fun constArrayStoreSelectAtSymbolicIndexIsSat() =
        assertEquals(KSolverStatus.SAT, solve(Shape.CONST_ARRAY_STORE_SYMBOLIC_INDEX))

    /** `arrays-exp` is only set when the caller declares [KTheory.Array], or declares nothing. */
    @Test
    fun constArrayIsSupportedWhenOptimizedForArrayTheory() =
        assertEquals(
            KSolverStatus.SAT,
            solve(Shape.CONST_ARRAY_STORE_SYMBOLIC_INDEX, setOf(KTheory.Array, KTheory.BV))
        )

    @Test
    fun uninterpretedArrayBaseIsUnaffected() =
        assertEquals(KSolverStatus.SAT, solve(Shape.UNINTERPRETED_BASE))

    @Test
    fun constArrayWithoutStoreIsUnaffected() =
        assertEquals(KSolverStatus.UNSAT, solve(Shape.NO_STORE))

    @Test
    fun constArrayReadAtConcreteIndexIsUnaffected() =
        assertEquals(KSolverStatus.UNSAT, solve(Shape.CONCRETE_INDEX))

    private enum class Shape { CONST_ARRAY_STORE_SYMBOLIC_INDEX, UNINTERPRETED_BASE, NO_STORE, CONCRETE_INDEX }

    private companion object {
        private const val STORED_BYTE = 0xaa.toByte()

        private fun solve(shape: Shape, theories: Set<KTheory>? = null): KSolverStatus =
            KContext().use { ctx ->
                KCvc5Solver(ctx).use { solver ->
                    theories?.let { solver.configure { optimizeForTheories(it, quantifiersAllowed = false) } }
                    solver.assert(ctx.query(shape))
                    solver.check()
                }
            }

        private fun KContext.query(shape: Shape) = mkArraySort(bv64Sort, bv8Sort).let { sort ->
            val base: KExpr<ByteArraySort> = when (shape) {
                Shape.UNINTERPRETED_BASE -> sort.mkConst("memory")
                else -> mkArrayConst(sort, mkBv(0.toByte()))
            }
            val array = if (shape == Shape.NO_STORE) base else mkArrayStore(base, mkBv(0L), mkBv(STORED_BYTE))
            val index = if (shape == Shape.CONCRETE_INDEX) mkBv(1L) else bv64Sort.mkConst("index")
            mkArraySelect(array, index) eq mkBv(STORED_BYTE)
        }
    }
}
