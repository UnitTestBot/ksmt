package io.ksmt.test.benchmarks

import com.microsoft.z3.Context
import kotlinx.coroutines.async
import kotlinx.coroutines.joinAll
import kotlinx.coroutines.runBlocking
import org.junit.jupiter.api.AfterAll
import org.junit.jupiter.api.Disabled
import org.junit.jupiter.api.Timeout
import org.junit.jupiter.api.parallel.Execution
import org.junit.jupiter.api.parallel.ExecutionMode
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.transformer.KNonRecursiveTransformer
import io.ksmt.solver.z3.KZ3SMTLibParser
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KSort
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap
import java.util.concurrent.TimeUnit
import kotlin.io.path.Path
import kotlin.io.path.absolutePathString
import kotlin.io.path.writeLines
import kotlin.system.measureNanoTime

@Disabled
@Execution(ExecutionMode.SAME_THREAD)
@Timeout(10, unit = TimeUnit.SECONDS)
class Z3ConverterBenchmark : BenchmarksBasedTest() {

    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun measureFormulaSize(name: String, samplePath: Path) = ignoreExceptions {
        with(KContext()) {
            val assertions = KZ3SMTLibParser(this).parse(samplePath)
            val size = assertions.sumOf { FormulaSizeCalculator.size(it) }
            saveData(name, "size", "$size")
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun measureNativeAssertionTime(name: String, samplePath: Path) = ignoreExceptions {
        Context().use { ctx ->
            val assertions = ctx.parseSMTLIB2File(
                samplePath.absolutePathString(),
                emptyArray(),
                emptyArray(),
                emptyArray(),
                emptyArray()
            )
            val solver = ctx.mkSolver()

            // force solver initialization
            solver.push()

            val assertTime = measureNanoTime {
                assertions.forEach { solver.add(it) }
            }
            saveData(name, "native", "$assertTime")
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun measureNativeParsingAndAssertionTime(name: String, samplePath: Path) = ignoreExceptions {
        Context().use { ctx ->
            val solver = ctx.mkSolver()

            // force solver initialization
            solver.push()

            val assertAndParseTime = measureNanoTime {
                val assertions = ctx.parseSMTLIB2File(
                    samplePath.absolutePathString(),
                    emptyArray(),
                    emptyArray(),
                    emptyArray(),
                    emptyArray()
                )
                assertions.forEach { solver.add(it) }
            }
            saveData(name, "native_parse", "$assertAndParseTime")
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun measureKsmtAssertionTime(name: String, samplePath: Path) = ignoreExceptions {
        with(KContext()) {
            val assertions = KZ3SMTLibParser(this).parse(samplePath)
            KZ3Solver(this).use { solver ->

                // force solver initialization
                solver.push()

                val internalizeAndAssert = measureNanoTime {
                    assertions.forEach { solver.assert(it) }
                }
                saveData(name, "ksmt", "$internalizeAndAssert")
            }
        }
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("z3TestData")
    fun measureRunnerAssertionTime(name: String, samplePath: Path) = ignoreExceptions {
        with(KContext()) {
            val assertions = KZ3SMTLibParser(this).parse(samplePath)
            solverManager.createSolver(this, KZ3Solver::class).use { solver ->

                // force solver initialization
                solver.push()

                val internalizeAndAssert = measureNanoTime {
                    runBlocking {
                        assertions.map { expr ->
                            async { solver.assertAsync(expr) }
                        }.joinAll()
                    }
                }
                saveData(name, "runner", "$internalizeAndAssert")
            }
        }
    }


    private inline fun ignoreExceptions(block: () -> Unit) = try {
        block()
    } catch (t: Throwable) {
        System.err.println(t.toString())
    }

    private class FormulaSizeCalculator(ctx: KContext) : KNonRecursiveTransformer(ctx) {
        private var expressionCount = 0
        override fun <T : KSort> transformExpr(expr: KExpr<T>): KExpr<T> {
            expressionCount++
            return super.transformExpr(expr)
        }

        companion object {
            fun size(expr: KExpr<*>): Int =
                FormulaSizeCalculator(expr.ctx)
                    .also { it.apply(expr) }
                    .expressionCount
        }
    }

    companion object {
        private val data = ConcurrentHashMap<String, ConcurrentHashMap<String, String>>()

        private fun saveData(sample: String, type: String, value: String) {
            data.getOrPut(sample) { ConcurrentHashMap() }[type] = value
        }

        @AfterAll
        @JvmStatic
        fun saveData() {
            val headerRow = data.values.firstOrNull()?.keys?.sorted() ?: return
            val columns = listOf("Sample name") + headerRow
            val orderedData = listOf(columns) + data.map { (name, sampleData) ->
                val row = headerRow.map { sampleData[it] ?: "" }
                listOf(name) + row
            }
            val csvData = orderedData.map { it.joinToString(separator = ",") }
            Path("data.csv")
                .writeLines(csvData)
        }

        @JvmStatic
        fun z3TestData() = testData(::skipSlowSamples)

        // skip samples with slow native assert
        private fun skipSlowSamples(name: String): Boolean {
            if (name.startsWith("QF_ABV_try3")) return false
            if (name.startsWith("QF_ABV_try5")) return false
            if (name.startsWith("QF_ABV_testcase")) return false
            if (name in slowSamples) return false
            return true
        }

        private val slowSamples = setOf(
            "QF_ABV_ridecore-qf_abv-bug.smt2",
            "QF_ABV_noregions-fullmemite.stp.smt2",
            "QF_ABV_blaster-concrete.stp.smt2",
            "QF_ABV_blaster-wp.ir.3.simplified13.stp.smt2",
            "QF_ABV_blaster-wp.ir.3.simplified8.stp.smt2",
            "QF_ABV_blaster.stp.smt2",
            "QF_ABV_ff.stp.smt2",
            "QF_ABV_grep0084.stp.smt2",
            "QF_ABV_grep0095.stp.smt2",
            "QF_ABV_grep0106.stp.smt2",
            "QF_ABV_grep0117.stp.smt2",
            "QF_ABV_grep0777.stp.smt2",
            "AUFLIA_smt8061204852622600993.smt2",
            "AUFLIRA_cl5_nebula_init_0222.fof.smt2",
            "AUFLIRA_quaternion_ds1_inuse_0005.fof.smt2",
            "AUFLIRA_quaternion_ds1_inuse_0222.fof.smt2",
            "AUFLIRA_quaternion_ds1_symm_1198.fof.smt2",
            "AUFLIRA_quaternion_ds1_symm_1558.fof.smt2",
            "AUFLIRA_quaternion_ds1_symm_2906.fof.smt2",
            "AUFLIRA_thruster_init_0967.fof.smt2",
            "AUFLIRA_thruster_inuse_1134.fof.smt2",
            "AUFLIRA_thruster_symm_0369.fof.smt2",
            "AUFLIRA_thruster_symm_0946.fof.smt2",
            "AUFLIRA_thruster_symm_2979.fof.smt2",
            "LIA_ARI592=1.smt2",
            "QF_ABV_egt-1334.smt2",
            "QF_ABV_egt-2021.smt2",
            "QF_ABV_egt-2319.smt2",
            "QF_ABV_egt-4810.smt2",
            "QF_ABV_egt-7600.smt2",
            "QF_ABV_no_init_multi_member24.smt2",
            "QF_ABVFP_query.01403.smt2",
            "QF_ABVFP_query.03635.smt2",
            "QF_ABVFP_query.05867.smt2",
            "QF_ABVFP_query.06109.smt2",
            "QF_ABVFP_query.07672.smt2",
            "QF_ABVFP_query.08043.smt2",
            "QF_ABVFP_query.08657.smt2",
            "QF_BVFP_query.01217.smt2",
            "QF_BVFP_query.03449.smt2",
            "QF_BVFP_query.07486.smt2",
            "QF_BVFP_query.08173.smt2",
            "QF_FP_add-has-solution-15709.smt2",
            "QF_FP_add-has-solution-8348.smt2",
            "QF_FP_add-has-solution-8905.smt2",
            "QF_FP_div-has-no-other-solution-14323.smt2",
            "QF_FP_div-has-solution-12164.smt2",
            "QF_FP_fma-has-no-other-solution-13136.smt2",
            "QF_FP_gt-has-no-other-solution-6602.smt2",
            "QF_FP_lt-has-no-other-solution-10072.smt2",
            "QF_FP_lt-has-no-other-solution-10686.smt2",
            "QF_FP_lt-has-no-other-solution-4316.smt2",
            "QF_FP_lt-has-solution-5619.smt2",
            "QF_FP_min-has-no-other-solution-18171.smt2",
            "QF_FP_min-has-solution-10260.smt2",
            "QF_FP_mul-has-no-other-solution-10919.smt2",
            "QF_FP_rem-has-no-other-solution-5399.smt2",
            "QF_FP_rem-has-no-other-solution-6643.smt2",
            "QF_FP_rem-has-solution-5252.smt2",
            "QF_FP_sqrt-has-no-other-solution-16567.smt2",
            "QF_FP_sqrt-has-solution-1666.smt2",
            "QF_FP_sub-has-no-other-solution-12726.smt2",
            "QF_FP_sub-has-solution-19199.smt2",
            "QF_FP_toIntegral-has-no-other-solution-1029.smt2",
            "QF_FP_toIntegral-has-no-other-solution-1345.smt2",
            "QF_FP_toIntegral-has-no-other-solution-264.smt2",
            "QF_UFLIA_xs_28_38.smt2",
            "QF_ABV_ndes.smt2"
        )
    }
}
