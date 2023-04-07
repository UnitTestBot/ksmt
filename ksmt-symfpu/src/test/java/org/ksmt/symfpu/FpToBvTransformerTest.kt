package org.ksmt.symfpu

import org.junit.jupiter.api.Test
import org.ksmt.KContext
import org.ksmt.expr.*
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaSolver
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.*
import org.ksmt.utils.cast
import org.ksmt.utils.getValue
import kotlin.test.assertEquals
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
        with(KContext()) {
            val (a, b) = createTwoFp32Variables()
            block(a, b)
        }

    private inline fun <R> withContextAndFp128Variables(block: KContext.(KApp<KFp128Sort, *>, KApp<KFp128Sort, *>) -> R): R =
        with(KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)) {
            val a by mkFp128Sort()
            val b by mkFp128Sort()
            block(a, b)
        }

    private inline fun <R> withContextAndFp64Variables(block: KContext.(KApp<KFp64Sort, *>, KApp<KFp64Sort, *>) -> R): R =
        with(KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)) {
            val a by mkFp64Sort()
            val b by mkFp64Sort()
            block(a, b)
        }

    @Test
    fun testUnpackExpr() = withContextAndFp128Variables { a, _ ->
        testFpExpr(a, mapOf("a" to a))
    }

    @Test
    fun testUnpackExpr1() = withContextAndFp32Variables { _, _ ->
        testFpExpr(mkFp32(1.0f), mapOf("a" to mkFp32(1.0f)))
    }

    @Test
    fun testFpToBvEqExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBv64EqExpr() = withContextAndFp64Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b), mapOf("a" to a, "b" to b))
    }

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
    fun testFpToBvMult5Expr() = with(KContext()) {
        val a = mkFp32(Float.fromBits(0b0_00000000_00000000000000000000001)) // min
        val b = mkFp32(Float.fromBits(0b0_01111110_00000000000000000000000)) // 0.5
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMult5NegExpr() = with(KContext()) {
        val a = mkFp32(Float.fromBits(0b0_00000000_00000000000000000000001)) // min
        val b = mkFp32(Float.fromBits(0b1_01111110_00000000000000000000000.toInt())) // -0.5
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
        )
    }

    @Test
    fun testFpToBvMultToMinExpr() = with(KContext()) {
        val a = mkFp32(Float.fromBits(0b0_00000000_00000000000000000000001)) // min
        val b = mkFp32(Float.fromBits(0b1_01111110_00000000000000000000001.toInt())) // -0.5...1
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b), mapOf("a" to a, "b" to b)
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

    @Test
    fun testFpToBvMultFp16RNEExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        ) { _, e ->
            mkFpIsSubnormalExpr(e)
        }
    }


    @Test
    fun testFpToBvMultFp16RTNExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a, b),
            mapOf("a" to a, "b" to b),
        )
    }


    @Test
    fun testFpToBvMultFp16RTPExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvMultFp32RTP() = with(KContext()) {
        val a = mkFp32(Float.fromBits(0b00111111100000000000000000100010))
        val b = mkFp32(Float.fromBits(0b10000000000000000000000000000001.toInt()))
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvZZMultRoundTowardNegativeFp16Expr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), a, b),
            mapOf("a" to a, "b" to b),
        )
    }


    @Test
    fun testFpToBvZMultFp32UnderflowZeroExpr() = with(KContext()) {
        // should return -0
        val a = mkFp32(Float.fromBits(0b00000000000000000000000011100111)) // 3.24E-43
        val b = mkFp32(Float.fromBits(0b10000000000111111101111011010011.toInt())) // -2.926835E-39
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvZMultFp32UnderflowMinExpr() = with(KContext()) {
        // should return -min but returns -0
        val a = mkFp32(Float.fromBits(0b00000000001111011000100100000111))
        val b = mkFp32(Float.fromBits(0b10110100000001010010000001111110.toInt()))
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvZMultFp32UnderflowZero2Expr() = with(KContext()) {
        val a = mkFp32(Float.fromBits(0b00000000111011101110111111111011))
        val b = mkFp32(Float.fromBits(0b10000000000000001110001000000110.toInt()))
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        )
    }


    @Test
    fun testFpToBvMultFp16RNAExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }


    @Test
    fun testFpToBvMultFp16RNAExprasdf() = with(KContext()) {
//        a: FP (sign false) (5 11100) (11 1111100000),
//        b: FP (sign false) (5 10101) (11 0101100111)
        val a: KExpr<KFp16Sort> = mkFpFromBvExpr(mkBv(false).cast(), mkBv(0b01011, 5u), mkBv(0b1111100000, 10u))
        val b: KExpr<KFp16Sort> = mkFpFromBvExpr(mkBv(false).cast(), mkBv(0b00100, 5u), mkBv(0b0101100111, 10u))
        val bApp: KApp<KFp16Sort, *> = b.cast()
        println("a: $a, b: $b")
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a.cast(), "b" to bApp),
        )
    }

    @Test
    fun testFpToBvMultFp32RNAExpr() = with(KContext()) {
        val a by mkFp32Sort()
        val b by mkFp32Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvMultFp32RNAExprahh() = with(KContext()) {
        val i = Float.fromBits(0b11000000010011001001011001110001.toInt())
        val j = Float.fromBits(0b00111111101101100100010001000000)
        println("i: ${i.toBits().toUInt().toString(2)}")
        println("j: ${j.toBits().toUInt().toString(2)}")
        val a = mkFp32(Float.fromBits(0b110000000_10011001001011001110001.toInt()))
        println("a: $a")
        val b = mkFp32(Float.fromBits(0b00111111101101100100010001000000))
        println("b: $b")

        val aUfp = unpack(fp32Sort, mkBv(0b11000000010011001001011001110001.toInt(), 32u))
        unpack(fp32Sort, mkBv(0b11000000010011001001011001110001.toInt(), 32u))
        println("aUfp: $aUfp, ${aUfp.sign} ${aUfp.unbiasedExponent} ${aUfp.normalizedSignificand}")
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvMultFp16RTZExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    // 8.86 sec
    @Test
    fun testFpToBvZMultFp16SlowExpr() = with(KContext(simplificationMode = KContext.SimplificationMode.SIMPLIFY)) {
//        1 5 10
        val upper =
            mkFpBiased(signBit = false, biasedExponent = 0b10001, significand = 0b0001111111, sort = mkFp16Sort()) // 2
        val lower = mkFpBiased(
            signBit = false, biasedExponent = 0b10001, significand = 0b0000000001, sort = mkFp16Sort()
        ) // 1.9991
//0b0000000000 0b1111111111  0b10000 0b01111
        val a by fp16Sort
        val b by fp16Sort
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), a, b),
            mapOf("a" to a, "b" to b),
        ) { _, _ ->
            trueExpr
            mkAnd(
                mkFpLessOrEqualExpr(lower, a),
                mkFpLessOrEqualExpr(lower, b),
                mkFpLessOrEqualExpr(a, upper),
                mkFpLessOrEqualExpr(b, upper),
            )
        }
    }

//    @Test
//    fun testFpToBvMultFp32Expr() = withContextAndVariables { a, b ->
//        testFpExpr(
//            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive), a, b), mapOf("a" to a, "b" to b)
//        )
//    }

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


    @Test
    fun testFpToBvAddFp16RTNExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvAddFp16RNAExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvAddFp16RTZExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvAddFp16RTPExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvAddFp16RNEExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvSubFp16RTNExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpSubExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvSubFp16RNAExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpSubExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvSubFp16RTZExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpSubExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardZero), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvSubFp16RTPExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpSubExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardPositive), a, b),
            mapOf("a" to a, "b" to b),
        )
    }

    @Test
    fun testFpToBvSubFp16RNEExpr() = with(KContext()) {
        val a by mkFp16Sort()
        val b by mkFp16Sort()
        testFpExpr(
            mkFpSubExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b),
            mapOf("a" to a, "b" to b),
        )
    }


    @Test
    fun testFpToBvNegateExpr() = with(KContext()) {
        val a by mkFp16Sort()
        testFpExpr(
            mkFpNegationExpr(a),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpToBvAbsExpr() = with(KContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpAbsExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsNormalExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsNormalExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsSubnormalExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsSubnormalExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsZeroExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsZeroExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsInfExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsInfiniteExpr(a),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpToBvIsNaNExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsNaNExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsPositiveExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsPositiveExpr(a),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToBvIsNegativeExpr() = with(KContext()) {
        val a by mkFp32Sort()

        testFpExpr(
            mkFpIsNegativeExpr(a),
            mapOf("a" to a),
        )
    }

    private fun KContext.defaultRounding() = mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)

    @Test
    fun testFpToFpUpExpr() = with(KContext()) {
        val a by mkFp16Sort()
        testFpExpr(
            mkFpToFpExpr(mkFp128Sort(), defaultRounding(), a),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpToUBvUpExpr() = with(KContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpToBvExprNoSimplify(defaultRounding(), a, 32, false),
            mapOf("a" to a),
        ) { _, _ ->
            mkFpLessExpr(a, mkFp32(UInt.MAX_VALUE.toFloat())) and mkFpIsPositiveExpr(a)
        }
    }

    @Test
    fun testFpToSBvUpExpr() = with(KContext()) {
        val a by mkFp32Sort()
        testFpExpr(
            mkFpToBvExprNoSimplify(defaultRounding(), a, 32, true),
            mapOf("a" to a),
        ) { _, _ ->
            mkFpLessExpr(a, mkFp32(Int.MAX_VALUE.toFloat())) and mkFpLessExpr(mkFp32(Int.MIN_VALUE.toFloat()), a)
        }
    }

    @Test
    fun testBvToFpExpr() = with(KContext()) {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(fp32Sort, defaultRounding(), a.cast(), true),
            mapOf("a" to a),
        )
    }

    @Test
    fun testBvToFpUnsignedExpr() = with(KContext()) {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(fp32Sort, defaultRounding(), a.cast(), false),
            mapOf("a" to a),
        )
    }

    @Test
    fun testFpToFpDownExpr() = with(KContext()) {
        val a by mkFp128Sort()
        testFpExpr(
            mkFpToFpExpr(mkFp16Sort(), defaultRounding(), a),
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
            mapOf("a" to a, "b" to b),
            extraAssert = { _, _ ->
                trueExpr
            }
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
//        solver.assert(extraAssert(transformedExpr, exprToTransform))
        solver.assert(transformedExpr neq exprToTransform.cast())

        // check assertions satisfiability with timeout
        println("checking satisfiability...")
        val status =
            solver.checkWithAssumptions(listOf(extraAssert(transformedExpr, exprToTransform)), timeout = 200.seconds)
        println("status: $status")
        when (status) {
            KSolverStatus.SAT -> {
                val model = solver.model()
                val transformed = model.eval(transformedExpr)
                val baseExpr = model.eval(exprToTransform)


                println("transformed: ${unpackedString(transformed, model)}")
                println("exprToTrans: ${unpackedString(baseExpr, model)}")
                for ((name, expr) in printVars) {
                    val evalUnpacked = unpackedString(expr, model)
                    println("$name :: $evalUnpacked")
                }
            }

            KSolverStatus.UNKNOWN -> {
                println("STATUS == UNKNOWN")
                println(solver.reasonOfUnknown())
            }

            KSolverStatus.UNSAT -> {
                println("STATUS == UNSAT")
                val model = solver.unsatCore()
                println(model)
                model.forEach { println(it) }
            }
        }
        assertEquals(KSolverStatus.UNSAT, status)
    }

    @Test
    fun testFpToBvRoundToIntegralExpr() = with(KContext()) {
        val a by mkFp32Sort()
        val roundingModes = KFpRoundingMode.values()

        roundingModes.forEach {
            testFpExpr(
                mkFpRoundToIntegralExpr(mkFpRoundingModeExpr(it), a),
                mapOf("a" to a),
            )
        }
    }

    private fun KContext.unpackedString(value: KExpr<*>, model: KModel) = if (value.sort is KFpSort) {
        val sb = StringBuilder()
        val fpExpr: KExpr<KFpSort> = value.cast()
        val fpValue = model.eval(fpExpr) as KFpValue
        with(unpack(fpExpr.sort, mkFpToIEEEBvExpr(fpExpr.cast()))) {
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

            sb.append(" \nactually ${(fpValue as? KFp32Value)?.value ?: (fpValue as? KFp16Value)?.value}}")
//            val unpacked = unpack(fp32Sort, packedFloat)
// vs my ${(model.eval(unpacked.toFp()) as? KFp32Value)?.value}
            sb.toString()
        }
    } else {
        "${model.eval(value)}"
    }
}

