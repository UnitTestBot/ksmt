package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaKind
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.Native
import io.ksmt.sort.KArray2Sort
import io.ksmt.sort.KArraySort
import io.ksmt.sort.KBv32Sort
import io.ksmt.utils.getValue
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.milliseconds

class SolverTest {
    private val ctx = KContext()
    private val solver = KBitwuzlaSolver(ctx)

    @Test
    fun testAbortHandling() {
        assertFailsWith(BitwuzlaNativeException::class) {
            // Incorrect native expression with invalid term (0)
            Native.bitwuzlaMkTerm1(solver.bitwuzlaCtx.bitwuzla, BitwuzlaKind.BITWUZLA_KIND_AND, 0)
        }
    }

    @Test
    fun testUnsatCoreGeneration(): Unit = with(ctx) {
        val a by boolSort
        val b by boolSort
        val c by boolSort

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
        val a by boolSort
        val b by boolSort

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
        val a by boolSort
        val b by boolSort
        val c by boolSort

        solver.assert(!c)
        solver.assert(a or b)

        solver.push()

        solver.assertAndTrack(!a)

        solver.push()

        solver.assertAndTrack(!b)

        var status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        var core = solver.unsatCore()
        assertEquals(2, core.size)
        assertTrue(!a in core && !b in core)

        solver.pop()

        solver.assert(a)

        status = solver.check()
        assertEquals(KSolverStatus.UNSAT, status)
        core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(!a in core)

        solver.pop()

        status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testPushPopAndAssumptions(): Unit = with(ctx) {
        val a by boolSort
        solver.assert(a)
        solver.push()
        solver.assertAndTrack(!a)
        var status = solver.checkWithAssumptions(listOf(a))
        assertEquals(KSolverStatus.UNSAT, status)
        val core = solver.unsatCore()
        assertEquals(1, core.size)
        assertTrue(!a in core || a in core)
        solver.pop()
        status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testTimeout(): Unit = with(ctx) {
        val arrayBase by mkArraySort(mkBv32Sort(), mkBv32Sort())
        val x by mkBv32Sort()

        var array: KExpr<KArraySort<KBv32Sort, KBv32Sort>> = arrayBase
        for (i in 0..1024) {
            val v = mkBv((i xor 1024))
            array = array.store(mkBv(4198400 + i), v)
        }

        var xoredX: KExpr<KBv32Sort> = x
        for (i in 0..10000) {
            val selectedValue = array.select(mkBv(4198500 + i))
            xoredX = mkBvXorExpr(xoredX, selectedValue)
            xoredX = mkBvOrExpr(mkBvUnsignedDivExpr(xoredX, mkBv(3)), mkBvUnsignedRemExpr(xoredX, mkBv(3)))
        }
        val someRandomValue = mkBv(42)
        val assertion = xoredX eq someRandomValue

        solver.assert(assertion)

        val status = solver.check(timeout = 1.milliseconds)

        assertEquals(KSolverStatus.UNKNOWN, status)
    }

    @Test
    fun testFpSupport(): Unit = with(ctx) {
        val a by fp32Sort
        val expr = mkFpGreaterExpr(a, 0.0f.expr) and mkFpLessExpr(a, 10.0f.expr)
        solver.assert(expr)
        val status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testUninterpretedSortSupport(): Unit = with(ctx) {
        val aSort = mkUninterpretedSort("a")
        val bSort = mkUninterpretedSort("b")
        val aaArraySort = mkArraySort(aSort, aSort)
        val abArraySort = mkArraySort(aSort, bSort)

        val ax by aSort
        val ay by aSort

        val bx by bSort
        val by by bSort

        val aaArray by aaArraySort
        val abArray by abArraySort

        solver.assert(aaArray.select(ax) eq ay)
        solver.assert(aaArray.select(ay) eq ax)

        solver.assert(abArray.select(ax) eq bx)
        solver.assert(abArray.select(ay) eq by)
        solver.assert(bx neq by)

        val status = solver.check()
        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testArrayLambdaEq(): Unit = with(ctx) {
        val sort = mkArraySort(bv32Sort, bv32Sort)
        val arrayVar by sort

        val bias by bv32Sort
        val idx by bv32Sort
        val lambdaBody = mkBvAddExpr(idx, bias)
        val lambda = mkArrayLambda(idx.decl, lambdaBody)

        solver.assert(arrayVar eq lambda)
        val status = solver.check()

        assertEquals(KSolverStatus.SAT, status)
    }

    @Test
    fun testArrayLambdaSelect(): Unit = with(ctx) {
        val sort = mkArraySort(bv32Sort, bv32Sort)
        val arrayVar by sort

        val bias by bv32Sort
        val idx by bv32Sort
        val lambdaBody = arrayVar.select(mkBvAddExpr(idx, bias))
        val lambda = mkArrayLambda(idx.decl, lambdaBody)

        val selectIdx by bv32Sort
        val selectValue by bv32Sort
        val lambdaSelectValue by bv32Sort

        solver.assert(bias neq mkBv(0))
        solver.assert(selectValue eq arrayVar.select(selectIdx))
        solver.assert(lambdaSelectValue eq lambda.select(mkBvSubExpr(selectIdx, bias)))

        assertEquals(KSolverStatus.SAT, solver.check())

        solver.assert(lambdaSelectValue neq  selectValue)

        assertEquals(KSolverStatus.UNSAT, solver.check())
    }

    @Test
    fun testArray2Model(): Unit = with(ctx) {
        val sort = mkArraySort(bv32Sort, bv32Sort, bv32Sort)
        val a by sort
        val b by sort

        var expr: KExpr<KArray2Sort<KBv32Sort, KBv32Sort, KBv32Sort>> = a
        for (i in 0 until 10) {
            expr = expr.store(mkBv(i), mkBv(i), mkBv(i))
        }

        solver.assert(b eq expr)
        assertEquals(KSolverStatus.SAT, solver.check())

        val model = solver.model()

        val modelValue = model.eval(b eq expr)
        assertEquals(trueExpr, modelValue)
    }

    @Test
    fun testFunModel(): Unit = with(ctx) {
        val f = mkFuncDecl("f", bv32Sort, listOf(bv32Sort, bv32Sort))

        for (i in 0 until 10) {
            solver.assert(f.apply(listOf(mkBv(i), mkBv(i))) eq mkBv(i))
        }

        assertEquals(KSolverStatus.SAT, solver.check())

        val model = solver.model()

        assertEquals(10, model.interpretation(f)?.entries?.size)
    }

}
