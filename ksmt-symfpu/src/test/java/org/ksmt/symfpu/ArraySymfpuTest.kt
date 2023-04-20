package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.ksmt.KContext
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.solver.z3.KZ3SolverConfiguration
import org.ksmt.utils.getValue
import kotlin.time.Duration.Companion.seconds

class ArraySymfpuTest {
    class SymfpuZ3Solver(ctx: KContext) : SymfpuSolver<KZ3SolverConfiguration>(KZ3Solver(ctx), ctx)

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpArrayExpr() = with(createContext()) {
        val sort = mkArraySort(fp32Sort, fp32Sort)

        val array by sort
        val index by fp32Sort
        val value by fp32Sort
        val x by fp32Sort
        val y by fp32Sort
        val expr = (array.select(index) eq x) and (array.store(index, y) eq array)
        val transformer = FpToBvTransformer(this)
        SymfpuZ3Solver(this).use { solver ->
            solver.assert(expr)
            // check assertions satisfiability with timeout
            println("checking satisfiability...")
            val status = solver.check(timeout = 200.seconds)
            println(status)

            val model = solver.model()
            println("x= ${model.eval(x)}")
            println("y= ${model.eval(y)}")
        }
    }



//    @Execution(ExecutionMode.CONCURRENT)
//    @Test
//    fun testFpArray2Expr() = with(createContext()) {
//        val sort = mkArraySort(fp32Sort, fp32Sort)
//
//        val array by sort
//        val index by fp32Sort
//        val value by fp32Sort
//        val x by fp32Sort
//        val y by fp32Sort
////        val decl = mkFuncDecl("F", fp32Sort, listOf(fp32Sort))
//
//        val const = mkArrayConst(sort, value)
//        val const2 = mkArrayConst(sort, value)
////(and (= (select a1 x) x) (= (store a1 x y) a1))
//
//        val expr = (array.select(index) eq x) and (array.store(index, y) eq array)
//
//
//        val transformer = FpToBvTransformer(this)
//        SymfpuZ3Solver(this).use { solver ->
//            solver.assert(expr)
//
//            // check assertions satisfiability with timeout
//            println("checking satisfiability...")
//            val status = solver.check(timeout = 200.seconds)
//            println(status)
//
//        }
//
//    }
}