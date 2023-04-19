package org.ksmt.symfpu

import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.ksmt.KContext
import org.ksmt.expr.KApp
import org.ksmt.expr.KConst
import org.ksmt.expr.KExpr
import org.ksmt.expr.KFp16Value
import org.ksmt.expr.KFp32Value
import org.ksmt.expr.KFpRoundingMode
import org.ksmt.expr.KFpValue
import org.ksmt.expr.printer.BvValuePrintMode
import org.ksmt.expr.printer.PrinterParams
import org.ksmt.expr.transformer.KNonRecursiveTransformer
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFp128Sort
import org.ksmt.sort.KFp32Sort
import org.ksmt.sort.KFp64Sort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.utils.cast
import org.ksmt.utils.getValue
import org.ksmt.utils.uncheckedCast
import kotlin.time.Duration.Companion.seconds

//typealias Fp = KFp16Sort
typealias Fp = KFp32Sort

class FpToBvTransformerTest {
    private fun KContext.createTwoFp32Variables(): Pair<KApp<Fp, *>, KApp<Fp, *>> {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        return Pair(a, b)
    }

    private fun KContext.zero() = mkFpZero(false, Fp(this))
    private fun KContext.negativeZero() = mkFpZero(true, Fp(this))
    private inline fun <R> withContextAndFp32Variables(block: KContext.(KApp<KFp32Sort, *>, KApp<KFp32Sort, *>) -> R): R =
        with(KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY))) {
            val (a, b) = createTwoFp32Variables()
            block(a, b)
        }

    private inline fun <R> withContextAndFp128Variables(block: KContext.(KApp<KFp128Sort, *>, KApp<KFp128Sort, *>) -> R): R =
        with(KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY))) {
            val a by mkFp128Sort()
            val b by mkFp128Sort()
            block(a, b)
        }

    private inline fun <R> withContextAndFp64Variables(block: KContext.(KApp<KFp64Sort, *>, KApp<KFp64Sort, *>) -> R): R =
        with(KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY))) {
            val a by mkFp64Sort()
            val b by mkFp64Sort()
            block(a, b)
        }

    @Test
    fun testUnpackExpr() = with(KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY)) {
        val a by mkFp32Sort()
        testFpExpr(a, mapOf("a" to a))
    }


    @Test
    fun testFpToBvEqExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b), mapOf("a" to a, "b" to b)) { _, _ ->
            !(mkFpIsNaNExpr(a) or mkFpIsNaNExpr(b))
        }
    }

    @Test
    fun testFpToBv32EqExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToBv128EqExpr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLessExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLess32Expr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b)) // 450 ms vs 42 sec :: ~x100
    }

    @Test
    fun testFpToBvLess64Expr() = withContextAndFp64Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b)) // 450 ms vs 42 sec :: ~x100
    }

    @Test
    fun testFpToBvLess128Expr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b)) // 450 ms vs 42 sec :: ~x100
    }

    @Test
    fun testFpToBvLessOrEqualExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessOrEqualExpr(a, b), mapOf("a" to a, "b" to b)) // 124 ms vs
    }

    @Test
    fun testFpToBvGreaterExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpGreaterExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvGreaterOrEqualExpr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpGreaterOrEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    // filter results for min(a,b) = ±0 as it is not a failure
    private fun KContext.assertionForZeroResults() = { transformedExpr: KExpr<Fp>, exprToTransform: KExpr<Fp> ->
        (transformedExpr eq zero() and (exprToTransform eq negativeZero())).not() and (transformedExpr eq negativeZero() and (exprToTransform eq zero())).not()
    }

    @Test
    fun testFpToBvMinExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpMinExpr(a, b), mapOf("a" to a, "b" to b), extraAssert = assertionForZeroResults())
    }

    @Test
    fun testFpToBvMult2Expr() = with(createContext()) {
        val a = mkFp32(1.0f)
        val b = mkFp32(1.5f)
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMult4Expr() = with(createContext()) {
        val a = mkFp32(Float.fromBits(0b1_00100110_00100111111000011010000.toInt()))
        val b = mkFp32(Float.fromBits(0b0_11011001_10111011101000000000100))
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMult5Expr() = with(createContext()) {
        val a = mkFp32(Float.fromBits(0b0_00000000_00000000000000000000001)) // min
        val b = mkFp32(Float.fromBits(0b0_01111110_00000000000000000000000)) // 0.5
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMult5NegExpr() = with(createContext()) {
        val a = mkFp32(Float.fromBits(0b0_00000000_00000000000000000000001)) // min
        val b = mkFp32(Float.fromBits(0b1_01111110_00000000000000000000000.toInt())) // -0.5
        testFpExpr(
            mkFpMulExprNoSimplify(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMaxExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpMaxExpr(a, b), mapOf("a" to a, "b" to b), extraAssert = assertionForZeroResults())
    }

    @Test
    fun testFpToBvMinRecursiveExpr() = withContextAndFp32Variables { a, b ->
        val exprToTransform1 = mkFpMinExpr(a, mkFpMinExpr(a, b))
        val exprToTransform2 = mkFpMinExpr(a, exprToTransform1)

        testFpExpr(
            mkFpMinExpr(exprToTransform1, exprToTransform2),
            mapOf("a" to a, "b" to b),
            extraAssert = assertionForZeroResults()
        )
    }


    @Test
    fun testFpToBvMin128Expr() = withContextAndFp128Variables { a, b ->
        val sort = mkFp128Sort()
        val zero = mkFpZero(false, sort)
        val negativeZero = mkFpZero(true, sort)
//        1:45 min vs 25 sec
//        25 sec with
        testFpExpr(
            mkFpMinExpr(a, b),
            mapOf("a" to a, "b" to b),
            extraAssert = { transformedExpr, exprToTransform ->
                (transformedExpr eq zero and (exprToTransform eq negativeZero)).not() and (transformedExpr eq negativeZero and (exprToTransform eq zero)).not()
            }
        )
    }

    @Test
    fun testFpToBvMin64Expr() = withContextAndFp64Variables { a, b ->
        val sort = fp64Sort
        val zero = mkFpZero(false, sort)
        val negativeZero = mkFpZero(true, sort)

        testFpExpr(
            mkFpMinExpr(a, b),
            mapOf("a" to a, "b" to b),
            extraAssert = { transformedExpr, exprToTransform ->
                (transformedExpr eq zero and (exprToTransform eq negativeZero)).not() and (transformedExpr eq negativeZero and (exprToTransform eq zero)).not()
            }
        )
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToBvMultFp16RNAExpr() = with(createContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    private fun <T : KSort> KContext.testFpExpr(
        exprToTransform: KExpr<T>,
        printVars: Map<String, KApp<*, *>> = emptyMap(),
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>) = { _, _ -> trueExpr }
    ) {
        val transformer = FpToBvTransformer(this)

        if (System.getProperty("os.name") == "Mac OS X") {
            KZ3Solver(this)
        } else {
            KBitwuzlaSolver(this)
        }.use { solver ->
            checkTransformer(transformer, solver, exprToTransform, printVars, extraAssert)
        }
    }


//    @Test
//    fun testFpToBvFmaFp16RTNExpr() = with(KContext()) {
//        val a by mkFp16Sort()
//        val b by mkFp16Sort()
//        val c by mkFp16Sort()
//        testFpExpr(
//            mkFpFusedMulAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a, b, c),
//            mapOf("a" to a, "b" to b, "c" to c),
//        )
//    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToBvSqrtFp16RTNExpr() = with(createContext()) {
        val a by mkFp16Sort()
        testFpExpr(
            mkFpSqrtExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a),
            mapOf("a" to a),
        )
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToBvAddFp16RNAExpr() = with(createContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

//    @Test
//    fun testFpToBvRem16Expr() = with(KContext()) {
//        val a by mkFp16Sort()
//        val b by mkFp16Sort()
//        testFpExpr(
//            mkFpRemExpr(a, b),
//            mapOf("a" to a, "b" to b),
//        )
//    }



    @Test
    fun testFpToBvNegateExpr() = with(createContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpNegationExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvNothingExpr() = with(createContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            (a),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpToBvAbsExpr() = with(createContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpAbsExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsNormalExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsNormalExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsSubnormalExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsSubnormalExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsZeroExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsZeroExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsInfExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsInfiniteExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsInfInfExpr() = with(KContext(simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY, printerParams = PrinterParams(BvValuePrintMode.BINARY))) {
//        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsInfiniteExprNoSimplify(mkFpInf(true, mkFp32Sort()))
//            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsZeroInfExpr() = with(createContext()) {
//        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsInfiniteExprNoSimplify(mkFpZero(false, mkFp32Sort()))
//            mapOf("a" to a),
        )
    }


    @Test
    fun testFpToBvIsNaNExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsNaNExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsPositiveExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsPositiveExpr(a),
            mapOf("a" to a),
        )
    }
    @Test
    fun testFpToBvIsPositiveNaNExpr() = with(KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY))) {
//        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsPositiveExprNoSimplify(mkFpNaN(mkFp32Sort())),
            mapOf(),
        )
    }

    @Test
    fun testFpToBvIsNegativeExpr() = with(createContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsNegativeExpr(a),
            mapOf("a" to a),
        )
    }

    private fun KContext.defaultRounding() = mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)

    @Test
    fun testFpToFpUpExpr() = with(createContext()) {
        val a by mkFp16Sort()
        testFpExpr(
            mkFpToFpExpr(mkFp128Sort(), defaultRounding(), a),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpToUBvUpExpr() = with(createContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpToBvExprNoSimplify(defaultRounding(), a, 32, false),
            mapOf("a" to a),
        ) { _, _ ->
            mkFpLessExpr(a, mkFp32(UInt.MAX_VALUE.toFloat())) and mkFpIsPositiveExpr(a)
        }
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToSBvUpExpr() = with(createContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpToBvExprNoSimplify(defaultRounding(), a, 32, true),
            mapOf("a" to a),
        ) { _, _ ->
            mkFpLessExpr(a, mkFp32(Int.MAX_VALUE.toFloat())) and mkFpLessExpr(mkFp32(Int.MIN_VALUE.toFloat()), a)
        }
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testBvToFpExpr() = with(createContext()) {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(fp32Sort, defaultRounding(), a.cast(), true),
            mapOf("a" to a),
        )
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testBvToFpUnsignedExpr() = with(createContext()) {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(fp32Sort, defaultRounding(), a.cast(), false),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpFromBvExpr() = with(createContext()) {
        val sign by mkBv1Sort()
        val e by mkBv16Sort()
        val sig by mkBv16Sort()
        testFpExpr(
            mkFpFromBvExpr(sign.cast(), e.cast(), sig.cast()),
            mapOf("sign" to sign, "e" to e, "sig" to sig),
        )
    }

    @Test
    fun testFpToFpDownExpr() = with(createContext()) {
        val a by mkFp128Sort()
        testFpExpr(
            mkFpToFpExpr(mkFp16Sort(), defaultRounding(), a),
            mapOf("a" to a),
        )
    }
    @Test
    fun testIteExpr() = with(createContext()) {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        testFpExpr(
            mkIteNoSimplify(trueExpr, a, b),
            mapOf("a" to a),
        )
    }

    @Test
    fun testBvBoolFormulaExpr() = withContextAndFp64Variables { a, b ->
        val sort = fp64Sort
        val zero = mkFpZero(false, sort)
        mkFpZero(true, sort)

        val asq = mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, a)

        testFpExpr(
            mkFpLessExpr(asq, zero),
            mapOf("a" to a, "b" to b)
        )
    }

    private fun <T : KSort> KContext.checkTransformer(
        transformer: FpToBvTransformer,
        solver: KSolver<*>,
        exprToTransform: KExpr<T>,
        printVars: Map<String, KApp<*, *>>,
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>)
    ) {

        val applied = transformer.apply(exprToTransform)
        val transformedExpr: KExpr<T> = ((applied as? UnpackedFp<*>)?.toFp() ?: applied).cast()

        val testTransformer = TestTransformerUseBvs(this, transformer.mapFpToBv)
        val toCompare = testTransformer.apply(exprToTransform)


        solver.assert(!mkEqNoSimplify(transformedExpr, toCompare.cast()))

        // check assertions satisfiability with timeout
        println("checking satisfiability...")
        val status =
            solver.checkWithAssumptions(
                listOf(testTransformer.apply(extraAssert(transformedExpr, toCompare))),
                timeout = 200.seconds
            )
        println("status: $status")
        if (status == KSolverStatus.SAT) {
            val model = solver.model()
            val transformed = model.eval(transformedExpr)
            val baseExpr = model.eval(toCompare)

            println("transformed: ${unpackedString(transformed, model)}")
            println("exprToTrans: ${unpackedString(baseExpr, model)}")
            for ((name, expr) in printVars) {
                val ufp = transformer.mapFpToBv[expr.cast()]
                val evalUnpacked = unpackedString(ufp ?: expr, model)
                println("$name :: $evalUnpacked")
            }
        } else if (status == KSolverStatus.UNKNOWN) {
            println("STATUS == UNKNOWN")
            println(solver.reasonOfUnknown())
        }
//        assertEquals(KSolverStatus.UNSAT, status)
        assertNotEquals(KSolverStatus.SAT, status)
    }

    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToBvRoundToIntegralExpr() = with(createContext()) {
        val a by mkFp32Sort()
        val roundingModes = KFpRoundingMode.values()

        roundingModes.forEach {
            testFpExpr(
                mkFpRoundToIntegralExpr(mkFpRoundingModeExpr(it), a),
                mapOf("a" to a),
            )
        }
    }

    private fun createContext() = KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY))

    private fun KContext.unpackedString(value: KExpr<*>, model: KModel) = if (value.sort is KFpSort) {
        val sb = StringBuilder()
//        val fpExpr: KExpr<KFpSort> = value.cast()
//        val fpValue = model.eval(fpExpr) as KFpValue
        val fpExpr: KExpr<KFpSort> by lazy { value.uncheckedCast() }
        val ufp = if (value is UnpackedFp<*>) value else unpackBiased(fpExpr.sort, mkFpToIEEEBvExpr(fpExpr.cast()))
        val fpValue = model.eval(ufp.toFp()) as KFpValue
        with(ufp) {
            sb.append("uFP sign ")
            model.eval(sign).print(sb)
            sb.append(" ")
            model.eval(unbiasedExponent).print(sb)
            sb.append(" ")
            model.eval(normalizedSignificand).print(sb)

            //nan, inf, zero
            sb.append(" nan=")
            model.eval(isNaN).print(sb)
            sb.append(" inf=")
            model.eval(isInf).print(sb)
            sb.append(" isZero=")
            model.eval(isZero).print(sb)


            sb.append("\ntoFp: ")
            model.eval(toFp()).print(sb)

            val packedFloat = packToBv(this)
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
            sb.append(" \nactually ${(fpValue as? KFp32Value)?.value ?: (fpValue as? KFp16Value)?.value}}")
//            val unpacked = unpack(fp32Sort, packedFloat)
// vs my ${(model.eval(unpacked.toFp()) as? KFp32Value)?.value}
            sb.toString()
        }
    } else {
        "${model.eval(value)}"
    }
}


fun <T : KFpSort> KContext.fromPackedBv(it: KExpr<KBvSort>, sort: T): KExpr<T> {
    return unpackBiased(sort, it).toFp()
}

class TestTransformerUseBvs(ctx: KContext, private val mapFpToBv: Map<KExpr<KFpSort>, UnpackedFp<KFpSort>>) :
    KNonRecursiveTransformer(ctx) {
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = if (expr.sort is KFpSort) {
        val asFp: KConst<KFpSort> = expr.cast()
        mapFpToBv[asFp]!!.toFp().cast()
    } else expr
}
