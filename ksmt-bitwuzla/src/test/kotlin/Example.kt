import org.ksmt.KContext
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.sort.KArraySort
import kotlin.test.*

class Example {

    @Test
    @Suppress("USELESS_IS_CHECK")
    fun test() = with(KContext()) {
        val a = boolSort.mkConst("a")
        val b = mkFuncDecl("b", boolSort, listOf(boolSort))
        val c = boolSort.mkConst("c")
        val e = mkArraySort(boolSort, boolSort).mkConst("e")
        val solver = KBitwuzlaSolver(this)
        solver.assert(a)
//        solver.assert(
//            mkUniversalQuantifier(
//                !(c eq falseExpr and !(c eq a)) or b.apply(listOf(c)), listOf(c.decl)
//            )
//        )
//        solver.assert(
//            mkUniversalQuantifier(
//                !(c eq trueExpr) or !b.apply(listOf(c)), listOf(c.decl)
//            )
//        )
        solver.assert(e.select(a) eq trueExpr)
        val status = solver.check()
        assertEquals(status, KSolverStatus.SAT)
        val model = solver.model()
        val aValue = model.eval(a)
        val cValue = model.eval(c)
        val eValue = model.eval(e)
        assertEquals(aValue, trueExpr)
        assertEquals(cValue, c)
        assertTrue(eValue.sort is KArraySort<*, *>)
        val bInterp = model.interpretation(b)
        assertNotNull(bInterp)
        val detachedModel = model.detach()
        solver.close()
        assertFailsWith(IllegalStateException::class) { model.eval(a) }
        assertEquals(aValue, detachedModel.eval(a))
        assertEquals(cValue, detachedModel.eval(c))
        assertEquals(eValue, detachedModel.eval(e))
    }
}
