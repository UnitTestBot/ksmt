package io.ksmt.solver.maxsmt.test.sat

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.maxsmt.constraints.HardConstraint
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolver
import io.ksmt.solver.maxsmt.test.KMaxSMTBenchmarkBasedTest
import io.ksmt.solver.maxsmt.test.parseMaxSATTest
import io.ksmt.solver.maxsmt.test.utils.Solver
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.z3.KZ3Solver
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.MethodSource
import java.nio.file.Path
import kotlin.io.path.name
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlin.time.Duration.Companion.seconds

abstract class KMaxSATBenchmarkTest : KMaxSMTBenchmarkBasedTest {
    protected fun getSmtSolver(solver: Solver): KSolver<out KSolverConfiguration> = with(ctx) {
        return when (solver) {
            Solver.Z3 -> KZ3Solver(this)
            Solver.BITWUZLA -> KBitwuzlaSolver(this)
            Solver.CVC5 -> KCvc5Solver(this)
            Solver.YICES -> KYicesSolver(this)
            Solver.PORTFOLIO ->
                throw NotImplementedError("Portfolio solver for MaxSAT is not supported in tests")
        }
    }

    abstract fun getSolver(solver: Solver): KMaxSMTSolver<KSolverConfiguration>

    protected val ctx: KContext = KContext()
    private lateinit var maxSATSolver: KMaxSMTSolver<out KSolverConfiguration>

    private fun initSolver(solver: Solver) {
        maxSATSolver = getSolver(solver)
    }

    @AfterEach
    fun closeSolver() = maxSATSolver.close()

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSATTestData")
    fun maxSATZ3Test(name: String, samplePath: Path) {
        testMaxSATSolver(name, samplePath, Solver.Z3)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSATTestData")
    fun maxSATBitwuzlaTest(name: String, samplePath: Path) {
        testMaxSATSolver(name, samplePath, Solver.BITWUZLA)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSATTestData")
    fun maxSATCvc5Test(name: String, samplePath: Path) {
        testMaxSATSolver(name, samplePath, Solver.CVC5)
    }

    @ParameterizedTest(name = "{0}")
    @MethodSource("maxSATTestData")
    fun maxSATYicesTest(name: String, samplePath: Path) {
        testMaxSATSolver(name, samplePath, Solver.YICES)
    }

    private fun testMaxSATSolver(name: String, samplePath: Path, solver: Solver) = with(ctx) {
        val testData = maxSATTestNameToExpectedResult.find { it.first == samplePath.name }
        require(testData != null) { "Test [$name] expected result must be specified" }

        val constraints = parseMaxSATTest(samplePath, this)

        var sumOfSoftConstraintsWeights = 0u

        initSolver(solver)

        constraints.forEach {
            if (it is HardConstraint) {
                maxSATSolver.assert(it.expression)
            } else {
                maxSATSolver.assertSoft(it.expression, (it as SoftConstraint).weight)
                sumOfSoftConstraintsWeights += it.weight
            }
        }

        val maxSATResult = maxSATSolver.checkMaxSMT(60.seconds)
        val satConstraintsScore = maxSATResult.satSoftConstraints.sumOf { it.weight }
        val expectedSatConstraintsScore =
            sumOfSoftConstraintsWeights - maxSATTestNameToExpectedResult.first { it.first == samplePath.name }.second

        assertEquals(SAT, maxSATResult.hardConstraintsSatStatus, "Hard constraints must be SAT")
        assertTrue(maxSATResult.maxSMTSucceeded, "MaxSAT was not successful [$name]")
        assertTrue(maxSATResult.satSoftConstraints.isNotEmpty(), "Soft constraints size should not be 0")
        assertEquals(
            expectedSatConstraintsScore,
            satConstraintsScore.toULong(),
            "Soft constraints score was [$satConstraintsScore], " +
                    "but must be [$expectedSatConstraintsScore]",
        )
    }

    // Elements are pairs of test name and expected test result (cost equal to the excluded soft constraints sum).
    private val maxSATTestNameToExpectedResult = hashSetOf<Pair<String, ULong>>(
        "ae_5_20_ibmq-casablanca_7.wcnf" to 15u,
        "af-synthesis_stb_50_20_8.wcnf" to 120u,
        "af-synthesis_stb_50_40_0.wcnf" to 111u,
        "af-synthesis_stb_50_40_8.wcnf" to 117u,
        "af-synthesis_stb_50_60_3.wcnf" to 115u,
        "af-synthesis_stb_50_80_7.wcnf" to 115u,
        "af-synthesis_stb_50_80_8.wcnf" to 116u,
        "af-synthesis_stb_50_100_3.wcnf" to 104u,
        "af-synthesis_stb_50_120_3.wcnf" to 100u,
        "af-synthesis_stb_50_140_1.wcnf" to 127u,
        "af-synthesis_stb_50_140_3.wcnf" to 96u,
        "af-synthesis_stb_50_140_8.wcnf" to 113u,
        "af-synthesis_stb_50_160_5.wcnf" to 113u,
        "af-synthesis_stb_50_180_1.wcnf" to 130u,
        "af-synthesis_stb_50_200_4.wcnf" to 105u,
        "af-synthesis_stb_50_200_5.wcnf" to 102u,
        "af-synthesis_stb_50_200_6.wcnf" to 111u,
        "amazon.dimacs.wcnf" to 113575u,
        "ar-3.wcnf" to 43814u,
        "archlinux.dimacs.wcnf" to 11744u,
        "auc.cat_paths_60_100_0006.txt.wcnf" to 70929u,
        "auc.cat_paths_60_150_0007.txt.wcnf" to 100364u,
        "auc.cat_paths_60_170_0007.txt.wcnf" to 94165u,
        "auc.cat_paths_60_200_0009.txt.wcnf" to 157873u,
        "auc.cat_reg_60_110_0004.txt.wcnf" to 75602u,
        "auc.cat_reg_60_130_0001.txt.wcnf" to 104032u,
        "auc.cat_reg_60_160_0002.txt.wcnf" to 164897u,
        "auc.cat_sched_60_80_0005.txt.wcnf" to 34950u,
        "auc.cat_sched_60_90_0001.txt.wcnf" to 82847u,
        "auc.cat_sched_60_100_0003.txt.wcnf" to 34771u,
        "auc.cat_sched_60_120_0004.txt.wcnf" to 82385u,
        "auc.cat_sched_60_150_0001.txt.wcnf" to 155865u,
        "auc.cat_sched_60_150_0005.txt.wcnf" to 72866u,
        "auc.cat_sched_60_150_0009.txt.wcnf" to 58699u,
        "auc.cat_sched_60_160_0003.txt.wcnf" to 118883u,
        "auc.cat_sched_60_200_0005.txt.wcnf" to 172768u,
        "cap71.wcsp.wcnf" to 9326144u,
        "cap72.wcsp.wcnf" to 9777981u,
        "cap91.wcsp.wcnf" to 7966472u,
        "cap92.wcsp.wcnf" to 8547029u,
        "cap131.wcsp.wcnf" to 7934385u,
        "cap132.wcsp.wcnf" to 8514942u,
        "car.formula_0.8_2021_atleast_15_max-3_reduced_incomplete_adaboost_2.wcnf" to 232u,
        "causal_Meta_7_528.wcnf" to 55120u,
        "causal_n5_i5_N1000_uai13_log_int.wcnf" to 46030289u,
        "causal_n6_i1_N10000_uai14_log_int.wcnf" to 1510725680u,
        "causal_n6_i6_N1000_uai14_log_int.wcnf" to 126257527700u,
        "causal_n7_i8_N10000_uai14_log_int.wcnf" to 11486104693u,
        "causal_n7_i10_N1000_uai14_log_int.wcnf" to 3246397504u,
        "causal_Pigs_6_10000.wcnf" to 25539892u,
        "causal_Statlog_7_752.wcnf" to 380356u,
        "causal_Voting_7_435.wcnf" to 930263u,
        "causal_Water_7_380.wcnf" to 414473u,
        "comp04.wcnf" to 35u,
        "comp06.wcnf" to 27u,
        "CSG40-40-95.wcnf" to 8847u,
        "CSG60-60-88.wcnf" to 7714u,
        "CSGNaive60-60-53.wcnf" to 9829u,
        "CSGNaive70-70-91.wcnf" to 11177u,
        "dblp.dimacs.wcnf" to 25014u,
        "dim.brock800_3.clq.wcnf" to 1079u,
        "dim.c-fat200-1.clq.wcnf" to 14u,
        "dim.c-fat500-1.clq.wcnf" to 10u,
        "dim.san400_0.7_3.clq.wcnf" to 1201u,
        "dir.5.wcsp.dir.wcnf" to 261u,
        "dir.28.wcsp.dir.wcnf" to 270105u,
        "dir.54.wcsp.dir.wcnf" to 37u,
        "dir.404.wcsp.dir.wcnf" to 114u,
        "dir.408.wcsp.dir.wcnf" to 6228u,
        "dir.507.wcsp.dir.wcnf" to 27390u,
        "dir.509.wcsp.dir.wcnf" to 27390u,
        "dir.1403.wcsp.dir.wcnf" to 459246u,
        "dir.1502.wcsp.dir.wcnf" to 28042u,
        "dir.1506.wcsp.dir.wcnf" to 354517u,
        "drmx-am12-outof-40-ecardn-w.wcnf" to 28u,
        "drmx-am12-outof-40-esortn-w.wcnf" to 28u,
        "drmx-am16-outof-45-emtot-w.wcnf" to 29u,
        "drmx-am16-outof-45-eseqc-w.wcnf" to 29u,
        "drmx-am20-outof-50-ekmtot-w.wcnf" to 30u,
        "drmx-am20-outof-50-emtot-w.wcnf" to 30u,
        "drmx-am20-outof-50-eseqc-w.wcnf" to 30u,
        "drmx-am24-outof-55-emtot-w.wcnf" to 31u,
        "drmx-am24-outof-55-etot-w.wcnf" to 31u,
        "drmx-am28-outof-60-emtot-w.wcnf" to 32u,
        "drmx-am28-outof-60-eseqc-w.wcnf" to 32u,
        "drmx-am28-outof-60-etot-w.wcnf" to 32u,
        "drmx-am32-outof-70-ecardn-w.wcnf" to 38u,
        "drmx-am32-outof-70-ekmtot-w.wcnf" to 38u,
        "drmx-am32-outof-70-emtot-w.wcnf" to 38u,
        "drmx-am32-outof-70-etot-w.wcnf" to 38u,
        "eas.310-15.wcnf" to 30501u,
        "eas.310-28.wcnf" to 33151u,
        "eas.310-29.wcnf" to 34431u,
        "eas.310-33.wcnf" to 35463u,
        "eas.310-43.wcnf" to 33138u,
        "eas.310-44.wcnf" to 50871u,
        "eas.310-55.wcnf" to 21952u,
        "eas.310-74.wcnf" to 21867u,
        "eas.310-91.wcnf" to 37183u,
        "eas.310-93.wcnf" to 19146u,
        "eas.310-94.wcnf" to 26160u,
        "eas.310-95.wcnf" to 25854u,
        "eas.310-97.wcnf" to 35249u,
        "ebay.dimacs.wcnf" to 123941u,
        "f1-DataDisplay_0_order4.seq-A-2-1-EDCBAir.wcnf" to 6223203u,
        "f1-DataDisplay_0_order4.seq-A-2-2-abcdeir.wcnf" to 481429u,
        "f1-DataDisplay_0_order4.seq-A-2-2-irabcde.wcnf" to 2220415u,
        "f1-DataDisplay_0_order4.seq-A-3-1-EDCBAir.wcnf" to 6240245u,
        "f1-DataDisplay_0_order4.seq-A-3-1-irabcde.wcnf" to 5960556u,
        "f1-DataDisplay_0_order4.seq-A-3-2-irabcde.wcnf" to 5955300u,
        "f1-DataDisplay_0_order4.seq-B-2-2-abcdeir.wcnf" to 53533u,
        "f1-DataDisplay_0_order4.seq-B-2-2-irEDCBA.wcnf" to 2273346u,
        "f49-DC_TotalLoss.seq-A-2-1-abcdeir.wcnf" to 27698412327u,
        "f49-DC_TotalLoss.seq-A-2-1-irEDCBA.wcnf" to 14779649425u,
        "f49-DC_TotalLoss.seq-A-2-combined-irabcde.wcnf" to 14735114187u,
        "f49-DC_TotalLoss.seq-A-3-2-irEDCBA.wcnf" to 87222797189u,
        "f49-DC_TotalLoss.seq-B-2-2-abcdeir.wcnf" to 44321234u,
        "f49-DC_TotalLoss.seq-B-2-combined-EDCBAir.wcnf" to 83838199998u,
        "f49-DC_TotalLoss.seq-B-3-combined-EDCBAir.wcnf" to 117355113043u,
        "f49-DC_TotalLoss.seq-B-3-combined-irabcde.wcnf" to 87177360578u,
        "facebook1.dimacs.wcnf" to 45581u,
        "github.dimacs.wcnf" to 187405u,
        "graphstate_6_6_rigetti-agave_8.wcnf" to 6u,
        "grover-noancilla_4_52_rigetti-agave_8.wcnf" to 42u,
        "grover-v-chain_4_52_ibmq-casablanca_7.wcnf" to 27u,
        "guardian.dimacs.wcnf" to 160777u,
        "inst2.lp.sm-extracted.wcnf" to 97u,
        "inst10.lp.sm-extracted.wcnf" to 105u,
        "inst22.lp.sm-extracted.wcnf" to 180u,
        "instance1.wcnf" to 607u,
        "instance2.wcnf" to 828u,
        "ItalyInstance1.xml.wcnf" to 12u,
        "k50-18-30.rna.pre.wcnf" to 462u,
        "k50-21-38.rna.pre.wcnf" to 497u,
        "k100-14-38.rna.pre.wcnf" to 1953u,
        "k100-20-63.rna.pre.wcnf" to 2030u,
        "k100-38-60.rna.pre.wcnf" to 1878u,
        "k100-40-52.rna.pre.wcnf" to 1861u,
        "k100-73-76.rna.pre.wcnf" to 2008u,
        "k100-78-85.rna.pre.wcnf" to 1744u,
        "lisbon-wedding-1-18.wcnf" to 961u,
        "lisbon-wedding-2-18.wcnf" to 1137u,
        "lisbon-wedding-3-17.wcnf" to 1035u,
        "lisbon-wedding-4-18.wcnf" to 803u,
        "lisbon-wedding-5-17.wcnf" to 802u,
        "lisbon-wedding-9-17.wcnf" to 394u,
        "lisbon-wedding-10-17.wcnf" to 377u,
        "log.8.wcsp.log.wcnf" to 2u,
        "log.28.wcsp.log.wcnf" to 270105u,
        "log.408.wcsp.log.wcnf" to 6228u,
        "log.505.wcsp.log.wcnf" to 21253u,
        "log.1401.wcsp.log.wcnf" to 459106u,
        "londonist.dimacs.wcnf" to 70703u,
        "metro_8_8_5_20_10_6_500_1_0.lp.sm-extracted.wcnf" to 82u,
        "metro_8_8_5_20_10_6_500_1_7.lp.sm-extracted.wcnf" to 89u,
        "metro_8_8_5_20_10_6_500_1_9.lp.sm-extracted.wcnf" to 105u,
        "metro_9_8_7_22_10_6_500_1_1.lp.sm-extracted.wcnf" to 52u,
        "metro_9_8_7_22_10_6_500_1_2.lp.sm-extracted.wcnf" to 60u,
        "metro_9_8_7_22_10_6_500_1_3.lp.sm-extracted.wcnf" to 44u,
        "metro_9_8_7_30_10_6_500_1_5.lp.sm-extracted.wcnf" to 47u,
        "metro_9_8_7_30_10_6_500_1_6.lp.sm-extracted.wcnf" to 31u,
        "metro_9_8_7_30_10_6_500_1_7.lp.sm-extracted.wcnf" to 47u,
        "metro_9_8_7_30_10_6_500_1_8.lp.sm-extracted.wcnf" to 55u,
        "metro_9_9_10_35_13_7_500_2_7.lp.sm-extracted.wcnf" to 37u,
        "MinWidthCB_milan_100_12_1k_1s_2t_3.wcnf" to 109520u,
        "MinWidthCB_milan_200_12_1k_4s_1t_4.wcnf" to 108863u,
        "MinWidthCB_mitdbsample_100_43_1k_2s_2t_2.wcnf" to 38570u,
        "MinWidthCB_mitdbsample_100_64_1k_2s_1t_2.wcnf" to 66045u,
        "MinWidthCB_mitdbsample_200_43_1k_2s_2t_2.wcnf" to 50615u,
        "MinWidthCB_mitdbsample_200_64_1k_2s_1t_2.wcnf" to 78400u,
        "MinWidthCB_mitdbsample_200_64_1k_2s_3t_2.wcnf" to 73730u,
        "MinWidthCB_mitdbsample_300_26_1k_3s_2t_3.wcnf" to 32420u,
        "MLI.ilpd_train_0_DNF_5_5.wcnf" to 700u,
        "mul.role_smallcomp_multiple_0.3_6.wcnf" to 139251u,
        "mul.role_smallcomp_multiple_1.0_6.wcnf" to 295598u,
        "openstreetmap.dimacs.wcnf" to 65915u,
        "pac.80cfe9a6-9b1b-11df-965e-00163e46d37a_l1.wcnf" to 1924238u,
        "pac.fa3d0fb2-db9e-11df-a0ec-00163e3d3b7c_l1.wcnf" to 4569599u,
        "pac.rand179_l1.wcnf" to 493118u,
        "pac.rand892_l1.wcnf" to 224702u,
        "pac.rand984_l1.wcnf" to 345082u,
        "ped2.B.recomb1-0.01-2.wcnf" to 7u,
        "ped2.B.recomb1-0.10-7.wcnf" to 588u,
        "ped3.D.recomb10-0.20-12.wcnf" to 349u,
        "ped3.D.recomb10-0.20-14.wcnf" to 7u,
        "portfoliovqe_4_18_rigetti-agave_8.wcnf" to 33u,
        "power-distribution_1_2.wcnf" to 3u,
        "power-distribution_1_4.wcnf" to 3u,
        "power-distribution_1_6.wcnf" to 3u,
        "power-distribution_1_8.wcnf" to 3u,
        "power-distribution_2_2.wcnf" to 10u,
        "power-distribution_2_8.wcnf" to 10u,
        "power-distribution_3_4.wcnf" to 1u,
        "power-distribution_7_6.wcnf" to 18u,
        "power-distribution_8_4.wcnf" to 40u,
        "power-distribution_8_7.wcnf" to 40u,
        "power-distribution_9_2.wcnf" to 18u,
        "power-distribution_11_6.wcnf" to 126u,
        "power-distribution_12_2.wcnf" to 216u,
        "power-distribution_12_5.wcnf" to 216u,
        "qaoa_4_16_ibmq-casablanca_7.wcnf" to 12u,
        "qft_5_26_ibmq-casablanca_7.wcnf" to 15u,
        "qftentangled_4_21_ibmq-casablanca_7.wcnf" to 15u,
        "qftentangled_4_39_rigetti-agave_8.wcnf" to 18u,
        "qftentangled_5_30_ibmq-london_5.wcnf" to 27u,
        "qftentangled_5_48_rigetti-agave_8.wcnf" to 24u,
        "qgan_6_15_ibmq-casablanca_7.wcnf" to 24u,
        "qpeexact_5_26_ibmq-casablanca_7.wcnf" to 15u,
        "qwalk-v-chain_3_30_ibmq-casablanca_7.wcnf" to 30u,
        "qwalk-v-chain_5_102_ibmq-london_5.wcnf" to 81u,
        "rail507.wcnf" to 174u,
        "rail516.wcnf" to 182u,
        "rail582.wcnf" to 211u,
        "ran.max_cut_60_420_2.asc.wcnf" to 703u,
        "ran.max_cut_60_420_5.asc.wcnf" to 715u,
        "ran.max_cut_60_420_9.asc.wcnf" to 674u,
        "ran.max_cut_60_500_2.asc.wcnf" to 900u,
        "ran.max_cut_60_560_3.asc.wcnf" to 1054u,
        "ran.max_cut_60_560_7.asc.wcnf" to 1053u,
        "ran.max_cut_60_600_1.asc.wcnf" to 1156u,
        "ran.max_cut_60_600_9.asc.wcnf" to 1149u,
        "random-dif-2.rna.pre.wcnf" to 929u,
        "random-dif-9.rna.pre.wcnf" to 456u,
        "random-dif-16.rna.pre.wcnf" to 768u,
        "random-dif-25.rna.pre.wcnf" to 512u,
        "random-net-20-5_network-4.net.wcnf" to 19602u,
        "random-net-30-3_network-2.net.wcnf" to 27606u,
        "random-net-30-4_network-3.net.wcnf" to 24925u,
        "random-net-40-2_network-8.net.wcnf" to 38289u,
        "random-net-40-2_network-9.net.wcnf" to 35951u,
        "random-net-40-3_network-5.net.wcnf" to 35488u,
        "random-net-40-4_network-2.net.wcnf" to 36427u,
        "random-net-50-3_network-5.net.wcnf" to 41356u,
        "random-net-50-4_network-8.net.wcnf" to 43243u,
        "random-net-60-3_network-3.net" to 50929u,
        "random-net-100-1_network-3.net.wcnf" to 91570u,
        "random-net-120-1_network-5.net.wcnf" to 117198u,
        "random-net-220-1_network-7.net.wcnf" to 203783u,
        "random-net-240-1_network-7.net.wcnf" to 219252u,
        "random-net-260-1_network-4.net.wcnf" to 238131u,
        "random-same-5.rna.pre.wcnf" to 456u,
        "random-same-12.rna.pre.wcnf" to 597u,
        "random-same-19.rna.pre.wcnf" to 337u,
        "random-same-25.rna.pre.wcnf" to 224u,
        "ran-scp.scp41_weighted.wcnf" to 429u,
        "ran-scp.scp48_weighted.wcnf" to 492u,
        "ran-scp.scp49_weighted.wcnf" to 641u,
        "ran-scp.scp51_weighted.wcnf" to 253u,
        "ran-scp.scp54_weighted.wcnf" to 242u,
        "ran-scp.scp56_weighted.wcnf" to 213u,
        "ran-scp.scp58_weighted.wcnf" to 288u,
        "ran-scp.scp65_weighted.wcnf" to 161u,
        "ran-scp.scp410_weighted.wcnf" to 514u,
        "ran-scp.scpnre5_weighted.wcnf" to 28u,
        "ran-scp.scpnrf1_weighted.wcnf" to 14u,
        "ran-scp.scpnrf4_weighted.wcnf" to 14u,
        "ran-scp.scpnrf5_weighted.wcnf" to 13u,
        "realamprandom_4_72_rigetti-agave_8.wcnf" to 36u,
        "role_smallcomp_0.7_11.wcnf" to 333834u,
        "role_smallcomp_0.75_8.wcnf" to 348219u,
        "role_smallcomp_0.85_4.wcnf" to 369639u,
        "role_smallcomp_0.85_7.wcnf" to 369639u,
        "Rounded_BTWBNSL_asia_100_1_3.scores_TWBound_2.wcnf" to 24564427u,
        "Rounded_BTWBNSL_asia_100_1_3.scores_TWBound_3.wcnf" to 24564427u,
        "Rounded_BTWBNSL_asia_10000_1_3.scores_TWBound_2.wcnf" to 2247208255u,
        "Rounded_BTWBNSL_hailfinder_100_1_3.scores_TWBound_2.wcnf" to 602126938u,
        "Rounded_BTWBNSL_hailfinder_100_1_3.scores_TWBound_3.wcnf" to 601946991u,
        "Rounded_BTWBNSL_Heart.BIC_TWBound_2.wcnf" to 239742296u,
        "Rounded_BTWBNSL_insurance_100_1_3.scores_TWBound_2.wcnf" to 170760179u,
        "Rounded_BTWBNSL_insurance_1000_1_3.scores_TWBound_2.wcnf" to 1389279780u,
        "Rounded_BTWBNSL_insurance_1000_1_3.scores_TWBound_3.wcnf" to 1388734978u,
        "Rounded_BTWBNSL_insurance_1000_1_3.scores_TWBound_4.wcnf" to 1388734978u,
        "Rounded_BTWBNSL_Water_1000_1_2.scores_TWBound_4.wcnf" to 1326306453u,
        "Rounded_CorrelationClustering_Ionosphere_BINARY_N200_D0.200.wcnf" to 4604640u,
        "Rounded_CorrelationClustering_Orl_BINARY_N320_D0.200.wcnf" to 4429109u,
        "Rounded_CorrelationClustering_Protein1_BINARY_N360.wcnf" to 27536228u,
        "Rounded_CorrelationClustering_Protein2_BINARY_N220.wcnf" to 13727551u,
        "Rounded_CorrelationClustering_Protein2_UNARY_N100.wcnf" to 3913145u,
        "simNo_1-s_15-m_50-n_50-fp_0.0001-fn_0.20.wcnf" to 11501657324586u,
        "simNo_2-s_5-m_100-n_100-fp_0.0001-fn_0.05.wcnf" to 99635408482313u,
        "simNo_3-s_5-m_50-n_50-fp_0.0001-fn_0.05.wcnf" to 18938961942919u,
        "simNo_5-s_15-m_100-n_100-fp_0.0001-fn_0.20.wcnf" to 113321765415159u,
        "simNo_6-s_5-m_100-n_50-fp_0.01-fn_0.05.wcnf" to 90981027155327u,
        "simNo_6-s_15-m_100-n_50-fp_0.01-fn_0.05.wcnf" to 60142712649443u,
        "simNo_8-s_5-m_100-n_100-fp_0.0001-fn_0.05.wcnf" to 74156301822200u,
        "simNo_8-s_5-m_100-n_100-fp_0.0001-fn_0.20.wcnf" to 131749300472480u,
        "simNo_9-s_5-m_100-n_100-fp_0.0001-fn_0.05.wcnf" to 131749300472480u,
        "simNo_10-s_15-m_100-n_50-fp_0.01-fn_0.20.wcnf" to 84803002848794u,
        "simNo_10-s_15-m_100-n_100-fp_0.0001-fn_0.20.wcnf" to 82981983123459u,
        "su2random_4_18_ibmq-casablanca_7.wcnf" to 24u,
        "su2random_5_30_ibmq-london_5.wcnf" to 51u,
        "tcp_students_91_it_2.wcnf" to 3024u,
        "tcp_students_91_it_3.wcnf" to 2430u,
        "tcp_students_91_it_6.wcnf" to 2877u,
        "tcp_students_91_it_7.wcnf" to 2505u,
        "tcp_students_91_it_13.wcnf" to 2730u,
        "tcp_students_98_it_8.wcnf" to 2727u,
        "tcp_students_98_it_9.wcnf" to 2469u,
        "tcp_students_98_it_12.wcnf" to 2994u,
        "tcp_students_105_it_7.wcnf" to 3024u,
        "tcp_students_105_it_13.wcnf" to 3360u,
        "tcp_students_105_it_15.wcnf" to 3258u,
        "tcp_students_112_it_1.wcnf" to 3513u,
        "tcp_students_112_it_3.wcnf" to 2916u,
        "tcp_students_112_it_5.wcnf" to 3366u,
        "tcp_students_112_it_7.wcnf" to 3513u,
        "tcp_students_112_it_15.wcnf" to 3585u,
        "test1--n-5000.wcnf" to 20u,
        "test2.wcnf" to 16u,
        "test2--n-5000.wcnf" to 3u,
        "test5--n-5000.wcnf" to 2u,
        "test9--n-5000.wcnf" to 2u,
        "test18--n-5000.wcnf" to 22u,
        "test25--n-10000.wcnf" to 4u,
        "test34--n-10000.wcnf" to 3u,
        "test41--n-15000.wcnf" to 5u,
        "test42--n-15000.wcnf" to 2u,
        "test53--n-15000.wcnf" to 10u,
        "test54--n-15000.wcnf" to 45u,
        "test66--n-20000.wcnf" to 1u,
        "test67--n-20000.wcnf" to 1u,
        "test70--n-20000.wcnf" to 5u,
        "test75--n-20000.wcnf" to 5u,
        "up-.mancoosi-test-i10d0u98-11.wcnf" to 1780771u,
        "up-.mancoosi-test-i10d0u98-16.wcnf" to 1780806u,
        "up-.mancoosi-test-i20d0u98-9.wcnf" to 1780788u,
        "up-.mancoosi-test-i30d0u98-3.wcnf" to 1780860u,
        "up-.mancoosi-test-i40d0u98-7.wcnf" to 1780807u,
        "up-.mancoosi-test-i40d0u98-17.wcnf" to 1780852u,
        "vio.role_smallcomp_violations_0.3_3.wcnf" to 185080u,
        "vio.role_smallcomp_violations_0.45_8.wcnf" to 244141u,
        "vqe_4_12_ibmq-casablanca_7.wcnf" to 15u,
        "vqe_5_20_ibmq-london_5.wcnf" to 33u,
        "warehouse0.wcsp.wcnf" to 328u,
        "warehouse1.wcsp.wcnf" to 730567u,
        "wcn.adult_train_3_DNF_1_5.wcnf" to 24254u,
        "wcn.ilpd_test_8_CNF_4_20.wcnf" to 287u,
        "wcn.ionosphere_train_5_DNF_2_10.wcnf" to 47u,
        "wcn.parkinsons_test_5_CNF_2_10.wcnf" to 58u,
        "wcn.pima_test_3_CNF_1_5.wcnf" to 125u,
        "wcn.tictactoe_test_8_CNF_2_20.wcnf" to 346u,
        "wcn.titanic_test_7_CNF_5_20.wcnf" to 557u,
        "wcn.titanic_test_8_DNF_1_20.wcnf" to 449u,
        "wcn.titanic_train_7_CNF_5_15.wcnf" to 3262u,
        "wcn.titanic_train_8_CNF_5_10.wcnf" to 2201u,
        "wcn.transfusion_test_7_DNF_3_5.wcnf" to 96u,
        "wcn.transfusion_train_2_CNF_5_10.wcnf" to 1600u,
        "wcn.transfusion_train_3_CNF_3_15.wcnf" to 2400u,
        "WCNF_pathways_p01.wcnf" to 2u,
        "WCNF_pathways_p03.wcnf" to 30u,
        "WCNF_pathways_p05.wcnf" to 60u,
        "WCNF_pathways_p06.wcnf" to 64u,
        "WCNF_pathways_p08.wcnf" to 182u,
        "WCNF_pathways_p09.wcnf" to 157u,
        "WCNF_pathways_p10.wcnf" to 129u,
        "WCNF_pathways_p12.wcnf" to 188u,
        "WCNF_pathways_p14.wcnf" to 207u,
        "WCNF_pathways_p16.wcnf" to 257u,
        "WCNF_storage_p02.wcnf" to 5u,
        "WCNF_storage_p06.wcnf" to 173u,
        "wei.SingleDay_3_weighted.wcnf" to 35439u,
        "wei.Subnetwork_7_weighted.wcnf" to 43213u,
        "wei.Subnetwork_9_weighted.wcnf" to 82813u,
        "wikipedia.dimacs.wcnf" to 42676u,
        "wpm.mancoosi-test-i1000d0u98-15.wcnf" to 92031744u,
        "wpm.mancoosi-test-i2000d0u98-25.wcnf" to 332548069u,
        "wpm.mancoosi-test-i3000d0u98-50.wcnf" to 422725765u,
        "wpm.mancoosi-test-i3000d0u98-70.wcnf" to 512958012u,
        "wpm.mancoosi-test-i4000d0u98-76.wcnf" to 738411504u,
        "youtube.dimacs.wcnf" to 227167u,
    )
}
