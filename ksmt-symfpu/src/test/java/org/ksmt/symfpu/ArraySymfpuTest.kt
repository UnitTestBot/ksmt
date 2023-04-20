package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.utils.getValue
import kotlin.time.Duration.Companion.seconds

class ArraySymfpuTest {

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpArrayExpr() = with(createContext()) {
        val sort = mkArraySort(fp32Sort, fp32Sort)

        val array by sort
        val index by fp32Sort
        val value by fp32Sort
        val x by fp32Sort
        val y by fp32Sort
        val decl = mkFuncDecl("F", fp32Sort, listOf(fp32Sort))

//        val mkConst = { mkArrayConst(sort, value) }
//        val mkSelect = { mkArraySelect(array, index) }

//        val mkStore = { mkArrayStore(array, index, value) }
//        val mkLambda = { mkArrayLambda(index.decl, mkArraySelect(array, index)) }
//        val mkAsArray = { mkFunctionAsArray(sort, decl) }
        val const = mkArrayConst(sort, value)
        val const2 = mkArrayConst(sort, value)
//(and (= (select a1 x) x) (= (store a1 x y) a1))

        val expr = (array.select(index) eq x) and (array.store(index, y) eq array)


        val transformer = FpToBvTransformer(this)
        KZ3Solver(this).use { solver ->


            val applied = transformer.apply(expr)
//            val applied = transformer.applyAndGetExpr(expr)

//        val transformedExpr: KExpr<T> = ((applied as? UnpackedFp<*>)?.toFp() ?: applied).cast()
//        val testTransformer = TestTransformerUseBvs(this, transformer.mapFpToBv)
//        val toCompare = testTransformer.apply(exprToTransform)


            solver.assert(applied)

            // check assertions satisfiability with timeout
            println("checking satisfiability...")
            val status = solver.check(timeout = 200.seconds)
            println(status)

        }

    }
}