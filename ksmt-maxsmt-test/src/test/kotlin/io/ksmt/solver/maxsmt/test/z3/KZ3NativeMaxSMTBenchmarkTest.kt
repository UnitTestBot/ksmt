package io.ksmt.solver.maxsmt.test.z3

import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Optimize
import com.microsoft.z3.Status
import io.github.oshai.kotlinlogging.KotlinLogging
import io.ksmt.KContext
import io.ksmt.solver.maxsmt.test.KMaxSMTBenchmarkBasedTest
import io.ksmt.solver.maxsmt.test.parseMaxSMTTestInfo
import io.ksmt.solver.maxsmt.test.statistics.JsonStatisticsHelper
import io.ksmt.solver.maxsmt.test.statistics.MaxSMTTestStatistics
import io.ksmt.solver.maxsmt.test.utils.Solver.Z3_NATIVE
import io.ksmt.solver.maxsmt.test.utils.getRandomString
import io.ksmt.solver.z3.KZ3Solver
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.BeforeAll
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.io.File
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.extension
import kotlin.system.measureTimeMillis
import kotlin.test.assertEquals

abstract class KZ3NativeMaxSMTBenchmarkTest : KMaxSMTBenchmarkBasedTest {
    private lateinit var z3Ctx: Context
    private lateinit var maxSMTSolver: Optimize
    private val logger = KotlinLogging.logger {}

    @BeforeEach
    fun initSolver() {
        z3Ctx = Context()
        maxSMTSolver = z3Ctx.mkOptimize()
    }

    @AfterEach
    fun close() {
        z3Ctx.close()
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSMTTestData")
    fun maxSMTZ3NativeTest(name: String, samplePath: Path) {
        testMaxSMTSolver(name, samplePath)
    }

    private fun testMaxSMTSolver(name: String, samplePath: Path) {
        val extension = "smt2"
        require(samplePath.extension == extension) {
            "File extension cannot be '${samplePath.extension}' as it must be $extension"
        }

        logger.info { "Test name: [$name]" }

        val testStatistics = MaxSMTTestStatistics(name, Z3_NATIVE)
        lateinit var expressions: List<BoolExpr>

        try {
            expressions = z3Ctx.parseSMTLIB2File(
                samplePath.toString(),
                emptyArray(),
                emptyArray(),
                emptyArray(),
                emptyArray(),
            ).toList()
        } catch (t: Throwable) {
            testStatistics.failedOnParsingOrConvertingExpressions = true
            testStatistics.exceptionMessage = t.message.toString()
            jsonHelper.appendTestStatisticsToFile(testStatistics)
            logger.error { t.message + System.lineSeparator() }
            throw t
        }

        val maxSmtTestIntoPath = samplePath.toString().removeSuffix(".smt2") + ".maxsmt"
        val maxSmtTestInfo = parseMaxSMTTestInfo(File(maxSmtTestIntoPath).toPath())

        val softConstraintsSize = maxSmtTestInfo.softConstraintsWeights.size

        val softExpressions =
            expressions.subList(
                expressions.lastIndex + 1 - softConstraintsSize,
                expressions.lastIndex + 1,
            )
        val hardExpressions =
            expressions.subList(0, expressions.lastIndex + 1 - softConstraintsSize)

        hardExpressions.forEach {
            maxSMTSolver.Assert(it)
        }

        var softConstraintsWeightsSum = 0u

        maxSmtTestInfo.softConstraintsWeights
            .zip(softExpressions)
            .forEach { (weight, expr) ->
                maxSMTSolver.AssertSoft(expr, weight.toInt(), "s")
                softConstraintsWeightsSum += weight
            }

        // Setting parameters (timeout).
        // Solver tries to find an optimal solution by default and suboptimal if timeout is set.
        // Cores are non-minimized by default.
        val params = z3Ctx.mkParams()
        // 1-minute timeout (in ms)
        params.add("timeout", 60000)
        // Choose an algorithm.
        params.add("maxsat_engine", "pd-maxres")
        // Prefer larger cores.
        params.add("maxres.hill_climb", true)
        params.add("maxres.max_core_size", 3)
        maxSMTSolver.setParameters(params)

        var maxSMTResult: Status?
        val elapsedTimeMs: Long

        try {
            elapsedTimeMs = measureTimeMillis {
                maxSMTResult = maxSMTSolver.Check()
            }
        } catch (ex: Exception) {
            testStatistics.exceptionMessage = ex.message.toString()
            jsonHelper.appendTestStatisticsToFile(testStatistics)
            logger.error { ex.message + System.lineSeparator() }
            throw ex
        }

        logger.info { "Elapsed time: $elapsedTimeMs ms --- MaxSMT call${System.lineSeparator()}" }
        testStatistics.elapsedTimeMs = elapsedTimeMs

        val actualSatSoftConstraintsWeightsSum = maxSmtTestInfo.softConstraintsWeights
            .zip(softExpressions)
            .fold(0uL) { acc, expr ->
                acc + if (maxSMTSolver.model.eval(expr.second, true).isTrue) expr.first.toULong() else 0uL
            }

        if (maxSMTResult == null || maxSMTResult == Status.UNKNOWN) {
            // TODO: ...
        }

        try {
            assertEquals(Status.SATISFIABLE, maxSMTResult, "MaxSMT returned $maxSMTResult status")
            assertEquals(
                maxSmtTestInfo.satSoftConstraintsWeightsSum,
                actualSatSoftConstraintsWeightsSum,
                "Soft constraints weights sum was [$actualSatSoftConstraintsWeightsSum], " +
                        "but must be [${maxSmtTestInfo.satSoftConstraintsWeightsSum}]",
            )
            testStatistics.passed = true
        } catch (ex: Exception) {
            logger.error { ex.message + System.lineSeparator() }
        } finally {
            jsonHelper.appendTestStatisticsToFile(testStatistics)
        }
    }

    companion object {
        init {
            KZ3Solver(KContext()).close()
        }

        private lateinit var jsonHelper: JsonStatisticsHelper

        @BeforeAll
        @JvmStatic
        fun initJsonHelper() {
            jsonHelper =
                JsonStatisticsHelper(
                    File(
                        "${
                            Paths.get("").toAbsolutePath()
                        }/src/test/resources/maxsmt-statistics-${getRandomString(16)}.json",
                    ),
                )
        }

        @AfterAll
        @JvmStatic
        fun closeJsonHelper() {
            jsonHelper.markLastTestStatisticsAsProcessed()
        }
    }
}
