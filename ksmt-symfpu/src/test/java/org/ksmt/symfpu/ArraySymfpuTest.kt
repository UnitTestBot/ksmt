package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.printer.BvValuePrintMode
import org.ksmt.expr.printer.PrinterParams
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.solver.z3.KZ3SolverConfiguration
import org.ksmt.utils.getValue
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
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(expr)
            // check assertions satisfiability with timeout
            println("checking satisfiability...")
            val status = solver.check(timeout = 200.seconds)
            assertEquals(KSolverStatus.SAT, status)
            println(status)

            val model = solver.model()
            println("x= ${model.eval(x)}")
            println("y= ${model.eval(y)}")
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

}
