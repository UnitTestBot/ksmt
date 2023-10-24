package io.ksmt.solver.runner

import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.BeforeEach
import io.ksmt.KContext
import io.ksmt.runner.generated.createInstance
import io.ksmt.runner.generated.models.SolverType
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.solver.z3.KZ3SolverConfiguration
import io.ksmt.utils.getValue
import io.ksmt.utils.mkConst
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertNotEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class SolverRunnerTest {
    private lateinit var context: KContext
    private lateinit var solver: KSolver<KZ3SolverConfiguration>

    @BeforeEach
    fun createNewEnvironment() {
        context = KContext()
        solver = solverManager.createSolver(context, KZ3Solver::class)
    }

    @AfterEach
    fun clearResources() {
        solver.close()
        context.close()
    }

    companion object {
        private lateinit var solverManager: KSolverRunnerManager

        @BeforeAll
        @JvmStatic
        fun initSolverManager() {
            solverManager = KSolverRunnerManager(workerPoolSize = 1)
        }

        @AfterAll
        @JvmStatic
        fun closeSolverManager() {
            solverManager.close()
        }
    }

    @Test
    fun testUnsatCoreGeneration(): Unit = with(context) {
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
    fun testUnsatCoreGenerationNoAssumptions(): Unit = with(context) {
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
    fun testPushPop(): Unit = with(context) {
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
    fun testTimeout(): Unit = with(context) {
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

        val status = solver.checkWithAssumptions(emptyList(), timeout = 1.milliseconds)
        assertEquals(KSolverStatus.UNKNOWN, status)
        assertTrue("timeout" in solver.reasonOfUnknown())
    }

    @Test
    fun testSolverConfiguration(): Unit = with(context) {
        val i by intSort
        val j by intSort

        val expr = (i gt j) or (i lt j)
        solver.assert(expr)

        solver.configure { setZ3Option("random_seed", 17) }
        val status1 = solver.check()
        assertEquals(KSolverStatus.SAT, status1)
        val model1 = solver.model()

        solver.configure { setZ3Option("random_seed", 42) }
        val status2 = solver.check()
        assertEquals(KSolverStatus.SAT, status2)
        val model2 = solver.model()

        solver.configure { setZ3Option("random_seed", 17) }
        val status3 = solver.check()
        assertEquals(KSolverStatus.SAT, status3)
        val model3 = solver.model()

        assertNotEquals(model1, model2)
        assertEquals(model1, model3)
    }

    @Test
    fun testBulkAssert(): Unit = with(context) {
        val a = boolSort.mkConst("a")
        val b = boolSort.mkConst("b")
        val c = boolSort.mkConst("c")

        val e1 = (a and b) or c
        val e2 = !(a and b)
        val e3 = !c

        solver.push()
        solver.assert(listOf(e1, e2, e3))
        assertEquals(KSolverStatus.UNSAT, solver.check())
        solver.pop()

        solver.push()
        solver.assertAndTrack(listOf(e1, e2, e3))
        assertEquals(KSolverStatus.UNSAT, solver.check())

        val core = solver.unsatCore()
        assertTrue(core.isNotEmpty())
    }

    @Test
    fun testSolverInstanceCreation(): Unit = with(context){
        SolverType.Z3.createInstance(this)
        SolverType.Bitwuzla.createInstance(this)
        SolverType.Yices.createInstance(this)
    }
}
