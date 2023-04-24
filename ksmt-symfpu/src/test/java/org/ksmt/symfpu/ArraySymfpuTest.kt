package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.printer.BvValuePrintMode
import org.ksmt.expr.printer.PrinterParams
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.solver.z3.KZ3SolverConfiguration
import org.ksmt.utils.getValue
import org.ksmt.utils.mkConst
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class ArraySymfpuTest {
    class SymfpuZ3Solver(ctx: KContext) : SymfpuSolver<KZ3SolverConfiguration>(KZ3Solver(ctx), ctx)

    @Test
    fun testFpArrayExpr() = with(KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)) {
        val sort = mkArraySort(fp32Sort, fp32Sort)

        val array by sort
        val index by fp32Sort
        val x by fp32Sort
        val y by fp32Sort
        val expr = (array.select(index) eq x) and (array.store(index, y) eq array)
        KZ3Solver(this).use { z3Solver->
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(expr)
            z3Solver.assert(expr)
            val status = solver.check(timeout = 200.seconds)
            val z3status = z3Solver.check(timeout = 200.seconds)
            assertEquals(KSolverStatus.SAT, status)
            assertEquals(KSolverStatus.SAT, z3status)

            val model = solver.model()
            val arrayEval = model.eval(array)
            println("array= ${arrayEval}")
        }
        }
    }

    @Test
    fun testFpRecursiveArrayExpr() = with(KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)) {
        val aSort = mkArraySort(fp32Sort, fp32Sort)
        val sort = mkArraySort(fp32Sort, aSort)

        val array by sort
        val aIndex by aSort
        val index by fp32Sort
        val expr = (array.select(index) eq aIndex)
//        and (array.store(index, aIndex) eq array)
        KZ3Solver(this).use { z3Solver->
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(expr)
            val status = solver.check(timeout = 200.seconds)
            assertEquals(KSolverStatus.SAT, status)

            val model = solver.model()
            val arrayEval = model.eval(array)
            println("array= ${arrayEval}")
        }
        }
    }

    @Test
    fun testFpArrayModelExpr() = with(KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)) {
        val sort = mkArraySort(fp32Sort, fp32Sort)

        val array by sort
        val expr = (array.select(0.0f.expr) eq 15.0f.expr) and (array.select(1.0f.expr) eq 30.0f.expr)
        KZ3Solver(this).use { z3Solver->
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(expr)
            val status = solver.check(timeout = 200.seconds)
            assertEquals(KSolverStatus.SAT, status)

            val model = solver.model()
            val arrayEval = model.eval(array)
            val zeroVal = model.eval(arrayEval.select(0.0f.expr))
            val oneVal = model.eval(arrayEval.select(1.0f.expr))
            val randVal = model.eval(arrayEval.select(2.0f.expr))
            println("array= ${arrayEval}")
            println("zero= ${zeroVal}")
            println("one= ${oneVal}")
            println("rand= ${randVal}")
        }
        }
    }


    @Test
    fun testFpArrayConstExpr() = with(KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY), simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)) {
        val sort = mkArraySort(fp32Sort, fp32Sort)

        val index by fp32Sort
        val value by fp32Sort

        val const = mkArrayConst(sort, value)
        val expr = const.select(index) neq value
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(expr)

            println("checking satisfiability...")
            val status = solver.check(timeout = 200.seconds)
            println(status)
            assertEquals(KSolverStatus.UNSAT, status)
        }
    }


    @Test
    fun testArrayLambdaEq(): Unit = with(KContext()) {
        val sort = mkArraySort(fp32Sort, fp32Sort)
        val arrayVar by sort

        val bias by fp32Sort
        val idx by fp32Sort
        val lambdaBody = mkFpAddExpr(defaultRounding(), idx, bias)

        val lambda = mkArrayLambda(idx.decl, lambdaBody)

        SymfpuZ3Solver(this).use { solver ->
            solver.assert(arrayVar eq lambda)
            val status = solver.check()

            assertEquals(KSolverStatus.SAT, status)
        }
    }

    @Test
    fun testArrayLambdaUnusedDeclEq(): Unit = with(KContext()) {
        val sort = mkArraySort(fp32Sort, fp32Sort)
        val arrayVar by sort

        val bias by fp32Sort
        val idx by fp32Sort
        val lambdaBody = mkFpAbsExpr(bias)

        val lambda = mkArrayLambda(idx.decl, lambdaBody)

        SymfpuZ3Solver(this).use { solver ->
            solver.assert(arrayVar eq lambda)
            val status = solver.check()

            assertEquals(KSolverStatus.SAT, status)
        }
    }

    @Test
    fun testArrayLambdaSelect(): Unit = with(KContext()) {
        val sort = mkArraySort(fp32Sort, fp32Sort)
        val arrayVar by sort

        val bias by fp32Sort
        val idx by fp32Sort
        val lambdaBody = arrayVar.select(mkFpAddExpr(defaultRounding(), idx, bias))
        val lambda = mkArrayLambda(idx.decl, lambdaBody)

        val selectIdx by fp32Sort
        val selectValue by fp32Sort
        val lambdaSelectValue by fp32Sort
        SymfpuZ3Solver(this).use { solver ->

            solver.assert(bias neq mkFp32(0.0f))
            solver.assert(selectValue eq arrayVar.select(selectIdx))
            solver.assert(lambdaSelectValue eq lambda.select(mkFpSubExpr(defaultRounding(), selectIdx, bias)))

            assertEquals(KSolverStatus.SAT, solver.check())

            solver.assert(lambdaSelectValue neq selectValue)

            assertEquals(KSolverStatus.SAT, solver.check()) // due to fp rounding
        }
    }


    @Test
    fun testUniversal(): Unit = with(KContext()) {
        val b = mkFuncDecl("b", boolSort, listOf(fp32Sort))
        val c = fp32Sort.mkConst("c")
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(
                mkUniversalQuantifier(
                    !(mkFpGreaterExpr(c, 0.0f.expr) and !(c eq 17.0f.expr)) or b.apply(listOf(c)), listOf(c.decl)
                )
            )

            assertEquals(KSolverStatus.SAT, solver.check())
        }
    }

}
