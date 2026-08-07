package io.ksmt.solver.z3

import com.microsoft.z3.Global
import io.ksmt.KContext
import io.ksmt.solver.KSolverStatus
import io.ksmt.utils.mkConst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class IncrementalApiTest {
    private val ctx = KContext()
    private val solver = KZ3Solver(ctx)

    @Test
    fun testUnsatCoreGeneration(): Unit = with(ctx) {
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")
        val c = boolSort.mkConst("c")

        val e1 = (a and b) or c
        val e2 = !(a and b)
        val e3 = !c

        solver.assert(e1)
        solver.assertAndTrack(e2)
        val status = solver.checkWithAssumptions(listOf(e3))
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(2, core.size)
        assertTrue(e2 in core)
        assertTrue(e3 in core)
    }

    @Test
    fun testUnsatCoreGenerationNoAssumptions(): Unit = with(ctx) {
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")

        val e1 = (a and b)
        val e2 = !(a and b)

        solver.assert(e1)
        solver.assertAndTrack(e2)
        val status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(e2 in core)
    }

    @Test
    fun testPushPop(): Unit = with(ctx) {
        val a = boolSort.mkConst("a")
        solver.assert(a)
        solver.push()
        solver.assertAndTrack(!a)
        var status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(!a in core)
        solver.pop()
        status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testTimeout(): Unit = with(ctx) {
        val array = mkArraySort(intSort, mkArraySort(intSort, intSort)).mkConst("array")
        val result = mkArraySort(intSort, intSort).mkConst("result")

        val i = intSort.mkConst("i")
        val j = intSort.mkConst("i")
        val idx = mkIte((i mod j) eq mkIntNum(100), i, j)
        val body = result.select(idx) eq array.select(i).select(j)
        val rule = mkUniversalQuantifier(body, listOf(i.decl, j.decl))
        solver.assert(rule)

        val x = intSort.mkConst("x")
        val queryBody = result.select(x) gt result.select(x + mkIntNum(10))
        val query = mkUniversalQuantifier(queryBody, listOf(x.decl))
        solver.assert(query)

        val status = withZ3MemoryLimit(MEMORY_LIMIT_MB) {
            solver.checkWithAssumptions(emptyList(), timeout = 1.milliseconds)
        }
        assertEquals(KSolverStatus.UNKNOWN, status)

        /**
         * Normally Z3 reports the timeout. If it happens to reach the memory limit before
         * its next timeout checkpoint it reports [Z3_OUT_OF_MEMORY_REASON] instead, which is
         * an equally valid "gave up on the resource limit" outcome.
         */
        val reason = solver.reasonOfUnknown()
        assertTrue(
            reason == TIMEOUT_REASON || reason == Z3_OUT_OF_MEMORY_REASON,
            "Unexpected reason of unknown: $reason"
        )
    }

    /**
     * Z3 only checks the [timeout] parameter at its internal checkpoints. On this query it
     * allocates memory far faster than it reaches one, so without a memory limit the check
     * grows to tens of gigabytes and the test process gets killed by the OS.
     *
     * The limit is a global Z3 parameter because a solver parameter would not survive:
     * every check replaces the solver parameters with the ones holding the timeout.
     */
    private inline fun <T> withZ3MemoryLimit(limitMb: Int, body: () -> T): T = try {
        Global.setParameter(Z3_MAX_MEMORY, limitMb.toString())
        body()
    } finally {
        Global.setParameter(Z3_MAX_MEMORY, UNLIMITED_MEMORY)
    }

    companion object {
        private const val Z3_MAX_MEMORY = "memory_max_size"
        private const val UNLIMITED_MEMORY = "0"
        private const val MEMORY_LIMIT_MB = 2048
        private const val TIMEOUT_REASON = "timeout"
        private const val Z3_OUT_OF_MEMORY_REASON = "out of memory"
    }
}
