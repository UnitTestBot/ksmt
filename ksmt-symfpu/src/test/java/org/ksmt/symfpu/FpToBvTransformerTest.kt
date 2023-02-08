package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.cast
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

//typealias Fp = KFp16Sort
typealias Fp = KFp32Sort

class FpToBvTransformerTest {
    private fun KContext.createTwoFpVariables(): Pair<KApp<Fp, *>, KApp<Fp, *>> {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        return Pair(a, b)
    }

    private fun KContext.zero() = mkFpZero(false, Fp(this))
    private fun KContext.negativeZero() = mkFpZero(true, Fp(this))
    private inline fun <R> withContextAndVariables(block: KContext.(KApp<Fp, *>, KApp<Fp, *>) -> R): R =
        with(KContext()) {
            val (a, b) = createTwoFpVariables()
            block(a, b)
        }

    @Test
    fun testUnpackExpr() = withContextAndVariables { a, _ ->
        testFpExpr(a, mapOf("a" to a))
    }

    @Test
    fun testUnpackExpr1() = withContextAndVariables { _, _ ->

        testFpExpr(mkFp32(1.0f), mapOf("a" to mkFp32(1.0f)))
    }

    @Test
    fun testFpToBvEqExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLessExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLessOrEqualExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpLessOrEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvGreaterExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpGreaterExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvGreaterOrEqualExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpGreaterOrEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    // filter results for min(a,b) = ±0 as it is not a failure
    private fun KContext.assertionForZeroResults() = { transformedExpr: KExpr<Fp>, exprToTransform: KExpr<Fp> ->
        (transformedExpr eq zero() and (exprToTransform eq negativeZero())).not() and (transformedExpr eq negativeZero() and (exprToTransform eq zero())).not()
    }

    @Test
    fun testFpToBvMinExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpMinExpr(a, b), mapOf("a" to a, "b" to b), extraAssert = assertionForZeroResults())
    }

    @Test
    fun testFpToBvMult2Expr() = with(KContext()) {
        val a = mkFp32(1.0f)
        val b = mkFp32(1.5f)
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMult4Expr() = with(KContext()) {
        val a = mkFp32(Float.fromBits(0b1_00100110_00100111111000011010000.toInt()))
        val b = mkFp32(Float.fromBits(0b0_11011001_10111011101000000000100))
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMult3Expr() = with(KContext()) {
        val a = mkFp32(1.0f)
        val b = mkFp32(1.0f)
//        val b = mkFp32(0.25f)
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMaxExpr() = withContextAndVariables { a, b ->
        testFpExpr(mkFpMaxExpr(a, b), mapOf("a" to a, "b" to b), extraAssert = assertionForZeroResults())
    }

    @Test
    fun testFpToBvMinRecursiveExpr() = withContextAndVariables { a, b ->
        val exprToTransform1 = mkFpMinExpr(a, mkFpMinExpr(a, b))
        val exprToTransform2 = mkFpMinExpr(a, exprToTransform1)

        testFpExpr(
            mkFpMinExpr(exprToTransform1, exprToTransform2),
            mapOf("a" to a, "b" to b),
            extraAssert = assertionForZeroResults()
        )
    }


    @Test
    fun testFpToBvZMultFp16Expr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
            extraAssert = assertionForUnderflowRounding(a.sort)
        )
    }

    private fun <Fp : KFpSort> KContext.assertionForUnderflowRounding(sort: Fp): (KExpr<Fp>, KExpr<Fp>) -> KExpr<KBoolSort> =
        {
                transformed: KExpr<Fp>,
                expr: KExpr<Fp>,
            ->
            !(mkFpIsZeroExpr(transformed) and (expr eq makeMin(sort, trueExpr).toFp() or (expr eq makeMin(
                sort,
                falseExpr
            ).toFp())))
        }

    @Test
    fun testFpToBvZMultFp32Expr() = withContextAndVariables { a, b ->
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
            assertionForUnderflowRounding(a.sort)
        )
    }

    private fun <T : KSort, Fp : KFpSort> KContext.testFpExpr(
        exprToTransform: KExpr<T>,
        printVars: Map<String, KApp<Fp, *>> = emptyMap(),
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>) = { _, _ -> trueExpr }
    ) {
        val transformer = FpToBvTransformer(this)

        KZ3Solver(this).use { solver ->
            checkTransformer(transformer, solver, exprToTransform, printVars, extraAssert)
        }
    }


    private fun <T : KSort, Fp : KFpSort> KContext.checkTransformer(
        transformer: FpToBvTransformer,
        solver: KZ3Solver,
        exprToTransform: KExpr<T>,
        printVars: Map<String, KApp<Fp, *>>,
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>)
    ) {
        val applied = transformer.apply(exprToTransform)
        val transformedExpr: KExpr<T> = ((applied as? UnpackedFp<*>)?.toFp() ?: applied).cast()
        solver.assert(extraAssert(transformedExpr, exprToTransform))
        solver.assert(transformedExpr neq exprToTransform.cast())

        // check assertions satisfiability with timeout
        val status = solver.check(timeout = 30000.seconds)

        if (status == KSolverStatus.SAT) {
            val model = solver.model()
            val transformed = model.eval(transformedExpr)
            val baseExpr = model.eval(exprToTransform)


            println("transformed: ${unpackedString(transformed, model)}")
            println("exprToTrans: ${unpackedString(baseExpr, model)}")
            for ((name, expr) in printVars) {
                val evalUnpacked = unpackedString(expr, model)
                println("$name :: $evalUnpacked")
            }
        } else if (status == KSolverStatus.UNKNOWN) {
            println(solver.reasonOfUnknown())
        }
        assertEquals(KSolverStatus.UNSAT, status)
    }

    private fun KContext.unpackedString(value: KExpr<*>, model: KModel) = if (value.sort is KFpSort) {
        val sb = StringBuilder()
        val fpExpr: KExpr<Fp> = value.cast()
        val fpValue = model.eval(fpExpr) as KFpValue
        with(unpack(fpExpr.sort, mkFpToIEEEBvExpr(fpExpr.cast()))) {
            sb.append("uFP sign ")
            model.eval(sign).print(sb)
            sb.append(" ")
            model.eval(exponent).print(sb)
            sb.append(" ")
            model.eval(significand).print(sb)

            //nan, inf, zero
            sb.append(" nan=")
            model.eval(isNaN).print(sb)
            sb.append(" inf=")
            model.eval(isInf).print(sb)
            sb.append(" zero=")
            model.eval(isZero).print(sb)


            sb.append("\ntoFp: ")
            model.eval(toFp()).print(sb)

            val packedFloat = mkFpToIEEEBvExpr(fpExpr)
            val pWidth = packedFloat.sort.sizeBits.toInt()
            val exWidth = sort.exponentBits.toInt()

            // Extract
            val packedSignificand = mkBvExtractExpr(pWidth - exWidth - 2, 0, packedFloat)
            val packedExponent = mkBvExtractExpr(pWidth - 2, pWidth - exWidth - 1, packedFloat)
            val sign = bvToBool(mkBvExtractExpr(pWidth - 1, pWidth - 1, packedFloat))
            sb.append("\nFP sign ")
            model.eval(sign).print(sb)
            sb.append(" ")
            model.eval(packedExponent).print(sb)
            sb.append(" ")
            model.eval(packedSignificand).print(sb)
            sb.append("\nbv: ")
            model.eval(packedFloat).print(sb)

            sb.append(" \nactually ${(fpValue as? KFp32Value)?.value}")

            sb.toString()
        }
    } else {
        "${model.eval(value)}"
    }
}


