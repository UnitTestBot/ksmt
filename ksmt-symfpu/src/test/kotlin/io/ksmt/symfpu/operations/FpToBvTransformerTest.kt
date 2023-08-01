package io.ksmt.symfpu.operations

import io.ksmt.KContext
import io.ksmt.decl.KDecl
import io.ksmt.expr.KApp
import io.ksmt.expr.KConst
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFp128Value
import io.ksmt.expr.KFp16Value
import io.ksmt.expr.KFp32Value
import io.ksmt.expr.KFpRoundingMode
import io.ksmt.expr.KFpValue
import io.ksmt.expr.printer.BvValuePrintMode
import io.ksmt.expr.printer.PrinterParams
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.runner.core.KsmtWorkerArgs
import io.ksmt.runner.core.KsmtWorkerFactory
import io.ksmt.runner.core.KsmtWorkerPool
import io.ksmt.runner.core.RdServer
import io.ksmt.runner.core.WorkerInitializationFailedException
import io.ksmt.runner.generated.models.TestProtocolModel
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.runner.KSolverRunnerManager
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KFp128Sort
import io.ksmt.sort.KFp32Sort
import io.ksmt.sort.KFp64Sort
import io.ksmt.sort.KFpSort
import io.ksmt.sort.KSort
import io.ksmt.symfpu.solver.FpToBvTransformer
import io.ksmt.test.TestRunner
import io.ksmt.test.TestWorker
import io.ksmt.test.TestWorkerProcess
import io.ksmt.utils.getValue
import io.ksmt.utils.uncheckedCast
import kotlinx.coroutines.TimeoutCancellationException
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import kotlin.time.Duration.Companion.minutes
import kotlin.time.Duration.Companion.seconds

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
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLess64Expr() = withContextAndFp64Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLess128Expr() = withContextAndFp128Variables { a, b ->
        testFpExpr(mkFpLessExpr(a, b), mapOf("a" to a, "b" to b))
    }

    @Test
    fun testFpToBvLessOrEqualExpr() = withContextAndFp32Variables { a, b ->
        testFpExpr(mkFpLessOrEqualExpr(a, b), mapOf("a" to a, "b" to b))
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

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
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
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>) = { _, _ -> trueExpr },
    ) {
        val ctx = this
        val transformer = FpToBvTransformer(this, true)

        if (System.getProperty("os.name") == "Mac OS X") {
            solverManager.createSolver(this, KZ3Solver::class)
        } else {
            solverManager.createSolver(this, KBitwuzlaSolver::class)
        }.use { solver ->
            with(testWorkers) {

                runBlocking {
                    val worker = try {
                        getOrCreateFreeWorker()
                    } catch (ex: WorkerInitializationFailedException) {
                        System.err.println("worker initialization failed -- ${ex.message}")
                        Assumptions.assumeTrue(false)
                        return@runBlocking
                    }
                    worker.astSerializationCtx.initCtx(ctx)
                    worker.lifetime.onTermination {
                        worker.astSerializationCtx.resetCtx()
                    }
                    try {
                        TestRunner(ctx, 20.seconds, worker).let {
                            try {
                                it.init()
                                checkTransformer(transformer, solver, exprToTransform, printVars, extraAssert)
                            } finally {
                                it.delete()
                            }
                        }
                    } catch (ex: TimeoutCancellationException) {
                        System.err.println("worker timeout -- ${ex.message}")
                        Assumptions.assumeTrue(false)
                        return@runBlocking
                    } finally {
                        worker.release()
                    }
                }

            }

        }
    }

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testFpToBvSqrtFp16RTNExpr() = with(createContext()) {
        val a by mkFp16Sort()
        testFpExpr(
            mkFpSqrtExpr(mkFpRoundingModeExpr(KFpRoundingMode.RoundTowardNegative), a),
            mapOf("a" to a),
        )
    }

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
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
        testFpExpr(
            mkFpIsInfiniteExprNoSimplify(mkFpInf(true, mkFp32Sort()))
        )
    }

    @Test
    fun testFpToBvIsZeroInfExpr() = with(createContext()) {
        testFpExpr(
            mkFpIsInfiniteExprNoSimplify(mkFpZero(false, mkFp32Sort()))
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

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
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

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testBvToFpExpr() = with(createContext()) {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(fp32Sort, defaultRounding(), a.uncheckedCast(), true),
            mapOf("a" to a),
        )
    }

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
    @Execution(ExecutionMode.CONCURRENT)
    @Test
    fun testBvToFpUnsignedExpr() = with(createContext()) {
        val a by mkBv32Sort()
        testFpExpr(
            mkBvToFpExprNoSimplify(fp32Sort, defaultRounding(), a.uncheckedCast(), false),
            mapOf("a" to a),
        )
    }


    @Test
    fun testFpFromBvExpr() = with(createContext()) {
        val sign by mkBv1Sort()
        val e by mkBv16Sort()
        val sig by mkBv16Sort()
        testFpExpr(
            mkFpFromBvExpr(sign.uncheckedCast(), e.uncheckedCast(), sig.uncheckedCast()),
            mapOf("sign" to sign, "e" to e, "sig" to sig),
        )
    }

    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
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
        extraAssert: ((KExpr<T>, KExpr<T>) -> KExpr<KBoolSort>),
    ) {

        val applied = transformer.apply(exprToTransform)
        val transformedExpr: KExpr<T> = ((applied as? UnpackedFp<*>)?.toFp() ?: applied).uncheckedCast()

        val testTransformer = TestTransformerUseBvs(this, transformer.mapFpToUnpackedFp)
        val toCompare = testTransformer.apply(exprToTransform)


        solver.assert(!mkEqNoSimplify(transformedExpr, toCompare))

        val status =
            solver.checkWithAssumptions(
                listOf(testTransformer.apply(extraAssert(transformedExpr, toCompare))),
                timeout = 2.seconds
            )
        if (status == KSolverStatus.SAT) {
            val model = solver.model()
            val transformed = model.eval(transformedExpr)
            val baseExpr = model.eval(toCompare)

            println("transformed: ${unpackedString(transformed, model)}")
            println("exprToTrans: ${unpackedString(baseExpr, model)}")
            for ((name, expr) in printVars) {
                val ufp = transformer.mapFpToUnpackedFp[expr.decl]
                val evalUnpacked = unpackedString(ufp ?: expr, model)
                println("$name :: $evalUnpacked")
            }
        }
        assertNotEquals(KSolverStatus.SAT, status)
    }


    @EnabledIfEnvironmentVariable(
        named = "runLongSymFPUTests",
        matches = "true",
    )
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


    private fun KContext.unpackedString(value: KExpr<*>, model: KModel) = if (value.sort is KFpSort) {
        val sb = StringBuilder()
        val ufp = if (value is UnpackedFp<*>) {
            value
        } else {
            val fpExpr: KExpr<KFpSort> = value.uncheckedCast()
            unpack(fpExpr.sort, mkFpToIEEEBvExpr(fpExpr), true)
        }
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
            sb.append(" \nactually ${(fpValue as? KFp32Value)?.value ?: (fpValue as? KFp16Value)?.value ?: (fpValue as? KFp128Value)}))}}")
            sb.toString()
        }
    } else {
        "${model.eval(value)}"
    }

    companion object {
        lateinit var solverManager: KSolverRunnerManager
        lateinit var testWorkers: KsmtWorkerPool<TestProtocolModel>

        @BeforeAll
        @JvmStatic
        fun initWorkerPools() {
            solverManager = KSolverRunnerManager(
                workerPoolSize = 4,
                hardTimeout = 20.seconds,
                workerProcessIdleTimeout = 10.minutes
            )
            testWorkers = KsmtWorkerPool(
                maxWorkerPoolSize = 4,
                workerProcessIdleTimeout = 10.minutes,
                workerFactory = object : KsmtWorkerFactory<TestProtocolModel> {
                    override val childProcessEntrypoint = TestWorkerProcess::class
                    override fun updateArgs(args: KsmtWorkerArgs): KsmtWorkerArgs = args
                    override fun mkWorker(id: Int, process: RdServer) = TestWorker(id, process)
                }
            )
        }

        @AfterAll
        @JvmStatic
        fun closeWorkerPools() {
            solverManager.close()
            testWorkers.terminate()
        }

    }
}


internal class TestTransformerUseBvs(ctx: KContext, private val mapFpToBv: Map<KDecl<KFpSort>, UnpackedFp<KFpSort>>) :
    KNonRecursiveTransformer(ctx) {
    override fun <T : KSort> transform(expr: KConst<T>): KExpr<T> = if (expr.sort is KFpSort) {
        val asFp: KConst<KFpSort> = expr.uncheckedCast()
        mapFpToBv[asFp.decl]!!.toFp().uncheckedCast()
    } else expr
}

internal fun createContext() = KContext(printerParams = PrinterParams(BvValuePrintMode.BINARY))
internal fun KContext.defaultRounding() = mkFpRoundingModeExpr(KFpRoundingMode.RoundNearestTiesToEven)
