package io.ksmt.symfpu

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpValue
import io.ksmt.expr.printer.BvValuePrintMode
import io.ksmt.expr.printer.PrinterParams
import io.ksmt.expr.rewrite.KExprUninterpretedDeclCollector
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.runner.KSolverRunnerManager
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp16Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.symfpu.operations.UnpackedFp
import io.ksmt.symfpu.operations.bvToBool
import io.ksmt.symfpu.operations.pack
import io.ksmt.symfpu.operations.packToBv
import io.ksmt.symfpu.operations.unpack
import io.ksmt.symfpu.solver.FpToBvTransformer
import io.ksmt.utils.getValue
import io.ksmt.utils.uncheckedCast
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.minutes

@Execution(ExecutionMode.CONCURRENT)
class FpToBvTransformerTest {
    @Test
    fun testUnpackExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(a)
    }

    @Test
    fun testFpToBvEqExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b)) { _, _ ->
            !(mkFpIsNaNExpr(a) or mkFpIsNaNExpr(b))
        }
    }

    @Test
    fun testFpToBv32EqExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b))
    }

    @Test
    fun testFpToBv128EqExpr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpEqualExpr(a, b))
    }

    @Test
    fun testFpToBvLessExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b))
    }

    @Test
    fun testFpToBvLess32Expr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b))
    }

    @Test
    fun testFpToBvLess64Expr() = withContextAndFp64Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b))
    }

    @Test
    fun testFpToBvLess128Expr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b))
    }

    @Test
    fun testFpToBvLessOrEqualExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessOrEqualExpr(a, b))
    }

    @Test
    fun testFpToBvGreaterExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpGreaterExpr(a, b))
    }

    @Test
    fun testFpToBvGreaterOrEqualExpr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpGreaterOrEqualExpr(a, b))
    }

    // filter results for min(a,b) = ±0 as it is not a failure
    private fun <Fp : KFpSort> KContext.assertionForZeroResults(sort: Fp) =
        { transformedExpr: KExpr<Fp>, exprToTransform: KExpr<Fp> ->
            val zero = mkFpZero(signBit = false, sort)
            val negativeZero = mkFpZero(signBit = true, sort)

            mkAnd(
                ((transformedExpr eq zero) and (exprToTransform eq negativeZero)).not(),
                ((transformedExpr eq negativeZero) and (exprToTransform eq zero)).not()
            )
        }

    @Test
    fun testFpToBvMinExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpMinExpr(a, b), assumption = assertionForZeroResults(a.sort))
    }


    @Test
    fun testFpToBvMaxExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpMaxExpr(a, b), assumption = assertionForZeroResults(a.sort))
    }

    @Test
    fun testFpToBvMinRecursiveExpr() = withContextAndFp32Variables { a, b ->
        val exprToTransform1 = mkFpMinExpr(a, mkFpMinExpr(a, b))
        val exprToTransform2 = mkFpMinExpr(a, exprToTransform1)

        testFpExpr(
            mkFpMinExpr(exprToTransform1, exprToTransform2),
            assumption = assertionForZeroResults(a.sort)
        )
    }

    @Test
    fun testFpToBvMin128Expr() = withContextAndFp128Variables { a, b ->
        testFpExpr(
            mkFpMinExpr(a, b),
            assumption = assertionForZeroResults(a.sort)
        )
    }

    @Test
    fun testFpToBvMin64Expr() = withContextAndFp64Variables { a, b ->
        testFpExpr(
            mkFpMinExpr(a, b),
            assumption = assertionForZeroResults(a.sort)
        )
    }

    @Test
    fun testFpToBvNegateExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpNegationExpr(a))
    }

    @Test
    fun testFpToBvAbsExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpAbsExpr(a))
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvAddExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b)
        )
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvSubExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(
            mkFpSubExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b)
        )
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvMulExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b)
        )
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvDivExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(
            mkFpDivExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, b)
        )
    }

    @Test
    fun testFpToBvIsNormalExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsNormalExpr(a))
    }

    @Test
    fun testFpToBvIsSubnormalExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsSubnormalExpr(a))
    }

    @Test
    fun testFpToBvIsZeroExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsZeroExpr(a))
    }

    @Test
    fun testFpToBvIsInfExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsInfiniteExpr(a))
    }

    @Test
    fun testFpToBvIsInfInfExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(
            mkFpIsInfiniteExprNoSimplify(mkFpInf(signBit = true, a.sort))
        )
    }

    @Test
    fun testFpToBvIsZeroInfExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(
            mkFpIsInfiniteExprNoSimplify(mkFpZero(signBit = false, a.sort))
        )
    }

    @Test
    fun testFpToBvIsNaNExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsNaNExpr(a))
    }

    @Test
    fun testFpToBvIsPositiveExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsPositiveExpr(a))
    }

    @Test
    fun testFpToBvIsPositiveNaNExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(
            mkFpIsPositiveExprNoSimplify(mkFpNaN(a.sort)),
        )
    }

    @Test
    fun testFpToBvIsNegativeExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(mkFpIsNegativeExpr(a))
    }

    @Test
    fun testFpToFpUpExpr() = withContextAndFp16Variables { a, _ ->
        testFpExpr(
            mkFpToFpExpr(mkFp128Sort(), mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a),
        )
    }

    @Test
    fun testFpToUBvUpExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(
            mkFpToBvExprNoSimplify(
                mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven),
                a,
                bvSize = 32,
                isSigned = false
            ),
            assumption = { _, _ ->
                mkFpLessExpr(a, mkFp32(UInt.MAX_VALUE.toFloat())) and mkFpIsPositiveExpr(a)
            },
        )
    }

    @Test
    fun testFpFromBvExpr() = withContext {
        val sign by mkBv1Sort()
        val e by mkBv16Sort()
        val sig by mkBv16Sort()
        testFpExpr(
            mkFpFromBvExpr(sign.uncheckedCast(), e.uncheckedCast(), sig.uncheckedCast()),
        )
    }

    @Test
    fun testIteExpr() = withContextAndFp32Variables { a, b ->
        val c by boolSort
        testFpExpr(mkIteNoSimplify(c, a, b))
    }

    @Test
    fun testBvBoolFormulaExpr() = withContextAndFp64Variables { a, _ ->
        val asq = mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a, a)
        testFpExpr(mkFpGreaterOrEqualExpr(asq, mkFpZero(signBit = false, fp64Sort)))
    }


    @Test
    fun testFpToBvRoundToIntegralExpr() = withContextAndFp32Variables { a, _ ->
        KFpRoundingMode.values().forEach {
            testFpExpr(mkFpRoundToIntegralExpr(mkFpRoundingModeExpr(it), a))
        }
    }

    @Test
    fun testFpToSBvUpExpr() = withContextAndFp32Variables { a, _ ->
        testFpExpr(
            mkFpToBvExprNoSimplify(
                mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven),
                a,
                bvSize = 32,
                isSigned = true
            ),
            assumption = { _, _ ->
                mkFpLessExpr(a, mkFp32(Int.MAX_VALUE.toFloat())) and mkFpLessExpr(mkFp32(Int.MIN_VALUE.toFloat()), a)
            },
        )
    }

    @Test
    fun testBvToFpExpr() = withContext {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(
                fp32Sort,
                mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven),
                a.uncheckedCast(),
                signed = true
            )
        )
    }

    @Test
    fun testFpToFpDownExpr() = withContextAndFp128Variables { a, _ ->
        testFpExpr(mkFpToFpExpr(mkFp16Sort(), mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven), a))
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testBvToFpUnsignedExpr() = withContext {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(
                fp32Sort,
                mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven),
                a.uncheckedCast(),
                signed = false
            )
        )
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvMultFp16RNAExpr() = withContextAndFp16Variables { a, b ->
        testFpExpr(
            mkFpMulExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
        )
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvSqrtFp16RTNExpr() = withContextAndFp16Variables { a, _ ->
        testFpExpr(
            mkFpSqrtExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a),
        )
    }

    @EnabledIfEnvironmentVariable(named = "runLongSymFpuTests", matches = "true")
    @Test
    fun testFpToBvAddFp16RNAExpr() = withContextAndFp16Variables { a, b ->
        testFpExpr(
            mkFpAddExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToAway), a, b),
        )
    }

    private fun <T : KSort> KContext.testFpExpr(
        exprToTransform: KExpr<T>,
        assumption: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort> = { _, _ -> trueExpr },
    ) {
        solverManager.createSolver(this, KZ3Solver::class).use { solver ->
            checkTransformer(solver, exprToTransform, assumption)
        }
    }

    private fun <T : KSort> KContext.checkTransformer(
        solver: KSolver<*>,
        originalExpr: KExpr<T>,
        assumption: (KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>,
    ) {
        val transformer = FpToBvTransformer(this, packedBvOptimization = true)
        val transformedExpr = transformer.applyUnpackFp(originalExpr)

        val testTransformer = TestTransformerUseBvs(this, transformer.mapFpToUnpackedFp)
        val expectedExpr = testTransformer.apply(originalExpr)

        // Check that expressions are equal under assumption
        val exprAssumption = assumption(transformedExpr, expectedExpr)
        val transformedAssumption = testTransformer.apply(exprAssumption)
        solver.assert(transformedAssumption)

        solver.assert(transformedExpr neq expectedExpr)

        val status = solver.check()

        if (status == KSolverStatus.SAT) {
            printDebugInfo(solver, transformedExpr, expectedExpr, transformer)
        }

        assertEquals(KSolverStatus.UNSAT, status)
    }

    private fun <T : KSort> KContext.printDebugInfo(
        solver: KSolver<*>,
        transformedExpr: KExpr<T>,
        expectedExpr: KExpr<T>,
        transformer: FpToBvTransformer
    ) {
        val model = solver.model()
        val transformed = model.eval(transformedExpr)
        val baseExpr = model.eval(expectedExpr)

        println("transformed: ${unpackedString(transformed, model)}")
        println("exprToTrans: ${unpackedString(baseExpr, model)}")

        KExprUninterpretedDeclCollector.collectUninterpretedDeclarations(expectedExpr).forEach { decl ->
            val ufp = transformer.mapFpToUnpackedFp[decl]
            val evalUnpacked = unpackedString(ufp ?: decl.apply(emptyList()), model)
            println("${decl.name} :: $evalUnpacked")
        }
    }

    private fun KContext.unpackedString(value: KExpr<*>, model: KModel): String {
        if (value.sort !is KFpSort) {
            return "${model.eval(value)}"
        }

        val ufp = if (value is UnpackedFp<*>) {
            value
        } else {
            val fpExpr: KExpr<KFpSort> = value.uncheckedCast()
            unpack(fpExpr.sort, mkFpToIEEEBvExpr(fpExpr), true)
        }
        val fpValue = model.eval(ufp.packToFp()) as KFpValue

        return buildString {
            append("uFP sign ")
            model.eval(ufp.sign).print(this)
            append(" ")
            model.eval(ufp.unbiasedExponent).print(this)
            append(" ")
            model.eval(ufp.normalizedSignificand).print(this)

            //nan, inf, zero
            append(" nan=")
            model.eval(ufp.isNaN).print(this)
            append(" inf=")
            model.eval(ufp.isInf).print(this)
            append(" isZero=")
            model.eval(ufp.isZero).print(this)


            append("\ntoFp: ")
            fpValue.print(this)

            val packedFloat = packToBv(ufp)
            val pWidth = packedFloat.sort.sizeBits.toInt()
            val exWidth = ufp.sort.exponentBits.toInt()

            // Extract
            val packedSignificand = mkBvExtractExpr(pWidth - exWidth - 2, 0, packedFloat)
            val packedExponent = mkBvExtractExpr(pWidth - 2, pWidth - exWidth - 1, packedFloat)
            val sign = bvToBool(mkBvExtractExpr(pWidth - 1, pWidth - 1, packedFloat))

            append("\nFP sign ")
            model.eval(sign).print(this)
            append(" ")
            model.eval(packedExponent).print(this)
            append(" ")
            model.eval(packedSignificand).print(this)
            append("\nbv: ")
            model.eval(packedFloat).print(this)
            append(" \nactually $fpValue))}}")
        }
    }

    private inline fun withContext(block: KContext.() -> Unit) {
        val ctx = KContext(
            printerParams = PrinterParams(BvValuePrintMode.BINARY),
            simplificationMode = KContext.SimplificationMode.NO_SIMPLIFY
        )
        return ctx.block()
    }

    private inline fun <Fp : KFpSort> withContextAndFpVariables(
        mkSort: KContext.() -> Fp,
        block: KContext.(KExpr<Fp>, KExpr<Fp>) -> Unit
    ) = withContext {
        val sort = mkSort()
        block(mkConst("a", sort), mkConst("b", sort))
    }

    private inline fun withContextAndFp16Variables(block: KContext.(KExpr<KFp16Sort>, KExpr<KFp16Sort>) -> Unit) =
        withContextAndFpVariables({ fp16Sort }, block)

    private inline fun withContextAndFp32Variables(block: KContext.(KExpr<KFp32Sort>, KExpr<KFp32Sort>) -> Unit) =
        withContextAndFpVariables({ fp32Sort }, block)

    private inline fun withContextAndFp64Variables(block: KContext.(KExpr<KFp64Sort>, KExpr<KFp64Sort>) -> Unit) =
        withContextAndFpVariables({ fp64Sort }, block)

    private inline fun withContextAndFp128Variables(block: KContext.(KExpr<KFp128Sort>, KExpr<KFp128Sort>) -> Unit) =
        withContextAndFpVariables({ mkFp128Sort() }, block)

    companion object {
        lateinit var solverManager: KSolverRunnerManager

        @BeforeAll
        @JvmStatic
        fun initWorkerPools() {
            solverManager = KSolverRunnerManager(
                workerPoolSize = 4,
                hardTimeout = 10.minutes,
                workerProcessIdleTimeout = 15.minutes
            )
        }

        @AfterAll
        @JvmStatic
        fun closeWorkerPools() {
            solverManager.close()
        }
    }
}

private class TestTransformerUseBvs(
    ctx: KContext,
    private val mapFpToBv: Map<KDecl<KFpSort>, UnpackedFp<KFpSort>>
) : KNonRecursiveTransformer(ctx) {
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> {
        if (expr.sort !is KFpSort) return expr

        val asFp = expr.uncheckedCast<_, KConst<KFpSort>>()
        val mappedExpr = mapFpToBv[asFp.decl]?.packToFp() ?: error("Expr was not mapped: $expr")
        return mappedExpr.uncheckedCast()
    }
}

private fun <T : KSort> FpToBvTransformer.applyUnpackFp(expr: KExpr<T>): KExpr<T> {
    val result = apply(expr)
    return if (result !is UnpackedFp<*>) result else result.packToFp().uncheckedCast()
}

private fun <Fp : KFpSort> UnpackedFp<Fp>.packToFp(): KExpr<Fp> =
    ctx.pack(ctx.packToBv(this), sort)

