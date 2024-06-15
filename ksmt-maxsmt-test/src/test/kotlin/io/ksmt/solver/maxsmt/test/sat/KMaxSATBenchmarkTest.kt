package io.ksmt.solver.maxsmt.test.sat

import io.ksmt.KContext
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverStatus.SAT
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.maxsmt.constraints.HardConstraint
import io.ksmt.solver.maxsmt.constraints.SoftConstraint
import io.ksmt.solver.maxsmt.solvers.KMaxSMTSolverBase
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

            Solver.Z3_NATIVE -> error("Unexpected solver type: Z3_NATIVE")
        }
    }

    abstract fun getSolver(solver: Solver): KMaxSMTSolverBase<KSolverConfiguration>

    protected val ctx: KContext = KContext()
    private lateinit var maxSATSolver: KMaxSMTSolverBase<out KSolverConfiguration>

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

        var sumOfSoftConstraintsWeights = 0uL

        initSolver(solver)

        constraints.forEach {
            if (it is HardConstraint) {
                maxSATSolver.assert(it.expression)
            } else {
                maxSATSolver.assertSoft(it.expression, (it as SoftConstraint).weight)
                sumOfSoftConstraintsWeights += it.weight
            }
        }

        val maxSATResult = maxSATSolver.checkMaxSMT(420.seconds)
        val satConstraintsScore = maxSATResult.satSoftConstraints.sumOf { it.weight.toULong() }
        val expectedSatConstraintsScore =
            sumOfSoftConstraintsWeights - maxSATTestNameToExpectedResult.first { it.first == samplePath.name }.second

        assertEquals(SAT, maxSATResult.hardConstraintsSatStatus, "Hard constraints must be SAT")
        assertTrue(
            !maxSATResult.timeoutExceededOrUnknown, "Timeout exceeded, solver returned UNKNOWN" +
                    "or something else happened [$name]"
        )
        assertTrue(maxSATResult.satSoftConstraints.isNotEmpty(), "Soft constraints size should not be 0")
        assertEquals(
            expectedSatConstraintsScore,
            satConstraintsScore,
            "Soft constraints score was [$satConstraintsScore], " +
                    "but must be [$expectedSatConstraintsScore]",
        )
    }

    // Elements are pairs of test name and expected test result (cost equal to the excluded soft constraints sum).
    private val maxSATTestNameToExpectedResult = hashSetOf(
        "ae_5_20_ibmq-casablanca_7.wcnf" to 15uL,
        "af-synthesis_stb_50_20_8.wcnf" to 120uL,
        "af-synthesis_stb_50_40_0.wcnf" to 111uL,
        "af-synthesis_stb_50_40_8.wcnf" to 117uL,
        "af-synthesis_stb_50_60_3.wcnf" to 115uL,
        "af-synthesis_stb_50_80_7.wcnf" to 115uL,
        "af-synthesis_stb_50_80_8.wcnf" to 116uL,
        "af-synthesis_stb_50_100_3.wcnf" to 104uL,
        "af-synthesis_stb_50_120_3.wcnf" to 100uL,
        "af-synthesis_stb_50_140_1.wcnf" to 127uL,
        "af-synthesis_stb_50_140_3.wcnf" to 96uL,
        "af-synthesis_stb_50_140_8.wcnf" to 113uL,
        "af-synthesis_stb_50_160_5.wcnf" to 113uL,
        "af-synthesis_stb_50_180_1.wcnf" to 130uL,
        "af-synthesis_stb_50_200_4.wcnf" to 105uL,
        "af-synthesis_stb_50_200_5.wcnf" to 102uL,
        "af-synthesis_stb_50_200_6.wcnf" to 111uL,
        "amazon.dimacs.wcnf" to 113575uL,
        "ar-3.wcnf" to 43814uL,
        "archlinux.dimacs.wcnf" to 11744uL,
        "auc.cat_paths_60_100_0006.txt.wcnf" to 70929uL,
        "auc.cat_paths_60_150_0007.txt.wcnf" to 100364uL,
        "auc.cat_paths_60_170_0007.txt.wcnf" to 94165uL,
        "auc.cat_paths_60_200_0009.txt.wcnf" to 157873uL,
        "auc.cat_reg_60_110_0004.txt.wcnf" to 75602uL,
        "auc.cat_reg_60_130_0001.txt.wcnf" to 104032uL,
        "auc.cat_reg_60_160_0002.txt.wcnf" to 164897uL,
        "auc.cat_sched_60_80_0005.txt.wcnf" to 34950uL,
        "auc.cat_sched_60_90_0001.txt.wcnf" to 82847uL,
        "auc.cat_sched_60_100_0003.txt.wcnf" to 34771uL,
        "auc.cat_sched_60_120_0004.txt.wcnf" to 82385uL,
        "auc.cat_sched_60_150_0001.txt.wcnf" to 155865uL,
        "auc.cat_sched_60_150_0005.txt.wcnf" to 72866uL,
        "auc.cat_sched_60_150_0009.txt.wcnf" to 58699uL,
        "auc.cat_sched_60_160_0003.txt.wcnf" to 118883uL,
        "auc.cat_sched_60_200_0005.txt.wcnf" to 172768uL,
        "cap71.wcsp.wcnf" to 9326144uL,
        "cap72.wcsp.wcnf" to 9777981uL,
        "cap91.wcsp.wcnf" to 7966472uL,
        "cap92.wcsp.wcnf" to 8547029uL,
        "cap131.wcsp.wcnf" to 7934385uL,
        "cap132.wcsp.wcnf" to 8514942uL,
        "car.formula_0.8_2021_atleast_15_max-3_reduced_incomplete_adaboost_2.wcnf" to 232uL,
        "causal_Meta_7_528.wcnf" to 55120uL,
        "causal_n5_i5_N1000_uai13_log_int.wcnf" to 46030289uL,
        "causal_n6_i1_N10000_uai14_log_int.wcnf" to 1510725680uL,
        "causal_n6_i6_N1000_uai14_log_int.wcnf" to 126257527700uL,
        "causal_n7_i8_N10000_uai14_log_int.wcnf" to 11486104693uL,
        "causal_n7_i10_N1000_uai14_log_int.wcnf" to 3246397504uL,
        "causal_Pigs_6_10000.wcnf" to 25539892uL,
        "causal_Statlog_7_752.wcnf" to 380356uL,
        "causal_Voting_7_435.wcnf" to 930263uL,
        "causal_Water_7_380.wcnf" to 414473uL,
        "comp04.wcnf" to 35uL,
        "comp06.wcnf" to 27uL,
        "CSG40-40-95.wcnf" to 8847uL,
        "CSG60-60-88.wcnf" to 7714uL,
        "CSGNaive60-60-53.wcnf" to 9829uL,
        "CSGNaive70-70-91.wcnf" to 11177uL,
        "dblp.dimacs.wcnf" to 25014uL,
        "dim.brock800_3.clq.wcnf" to 1079uL,
        "dim.c-fat200-1.clq.wcnf" to 14uL,
        "dim.c-fat500-1.clq.wcnf" to 10uL,
        "dim.san400_0.7_3.clq.wcnf" to 1201uL,
        "dir.5.wcsp.dir.wcnf" to 261uL,
        "dir.28.wcsp.dir.wcnf" to 270105uL,
        "dir.54.wcsp.dir.wcnf" to 37uL,
        "dir.404.wcsp.dir.wcnf" to 114uL,
        "dir.408.wcsp.dir.wcnf" to 6228uL,
        "dir.507.wcsp.dir.wcnf" to 27390uL,
        "dir.509.wcsp.dir.wcnf" to 36446uL,
        "dir.1403.wcsp.dir.wcnf" to 459246uL,
        "dir.1502.wcsp.dir.wcnf" to 28042uL,
        "dir.1506.wcsp.dir.wcnf" to 354517uL,
        "drmx-am12-outof-40-ecardn-w.wcnf" to 28uL,
        "drmx-am12-outof-40-esortn-w.wcnf" to 28uL,
        "drmx-am16-outof-45-emtot-w.wcnf" to 29uL,
        "drmx-am16-outof-45-eseqc-w.wcnf" to 29uL,
        "drmx-am20-outof-50-ekmtot-w.wcnf" to 30uL,
        "drmx-am20-outof-50-emtot-w.wcnf" to 30uL,
        "drmx-am20-outof-50-eseqc-w.wcnf" to 30uL,
        "drmx-am24-outof-55-emtot-w.wcnf" to 31uL,
        "drmx-am24-outof-55-etot-w.wcnf" to 31uL,
        "drmx-am28-outof-60-emtot-w.wcnf" to 32uL,
        "drmx-am28-outof-60-eseqc-w.wcnf" to 32uL,
        "drmx-am28-outof-60-etot-w.wcnf" to 32uL,
        "drmx-am32-outof-70-ecardn-w.wcnf" to 38uL,
        "drmx-am32-outof-70-ekmtot-w.wcnf" to 38uL,
        "drmx-am32-outof-70-emtot-w.wcnf" to 38uL,
        "drmx-am32-outof-70-etot-w.wcnf" to 38uL,
        "eas.310-15.wcnf" to 30501uL,
        "eas.310-28.wcnf" to 33151uL,
        "eas.310-29.wcnf" to 34431uL,
        "eas.310-33.wcnf" to 35463uL,
        "eas.310-43.wcnf" to 33138uL,
        "eas.310-44.wcnf" to 50871uL,
        "eas.310-55.wcnf" to 21952uL,
        "eas.310-74.wcnf" to 21867uL,
        "eas.310-91.wcnf" to 37183uL,
        "eas.310-93.wcnf" to 19146uL,
        "eas.310-94.wcnf" to 26160uL,
        "eas.310-95.wcnf" to 25854uL,
        "eas.310-97.wcnf" to 35249uL,
        "ebay.dimacs.wcnf" to 123941uL,
        "f1-DataDisplay_0_order4.seq-A-2-1-EDCBAir.wcnf" to 6223203uL,
        "f1-DataDisplay_0_order4.seq-A-2-2-abcdeir.wcnf" to 481429uL,
        "f1-DataDisplay_0_order4.seq-A-2-2-irabcde.wcnf" to 2220415uL,
        "f1-DataDisplay_0_order4.seq-A-3-1-EDCBAir.wcnf" to 6240245uL,
        "f1-DataDisplay_0_order4.seq-A-3-1-irabcde.wcnf" to 5960556uL,
        "f1-DataDisplay_0_order4.seq-A-3-2-irabcde.wcnf" to 5955300uL,
        "f1-DataDisplay_0_order4.seq-B-2-2-abcdeir.wcnf" to 53533uL,
        "f1-DataDisplay_0_order4.seq-B-2-2-irEDCBA.wcnf" to 2273346uL,
        "f49-DC_TotalLoss.seq-A-2-1-abcdeir.wcnf" to 27698412327uL,
        "f49-DC_TotalLoss.seq-A-2-1-irEDCBA.wcnf" to 14779649425uL,
        "f49-DC_TotalLoss.seq-A-2-combined-irabcde.wcnf" to 14735114187uL,
        "f49-DC_TotalLoss.seq-A-3-2-irEDCBA.wcnf" to 87222797189uL,
        "f49-DC_TotalLoss.seq-B-2-2-abcdeir.wcnf" to 44321234uL,
        "f49-DC_TotalLoss.seq-B-2-combined-EDCBAir.wcnf" to 83838199998uL,
        "f49-DC_TotalLoss.seq-B-3-combined-EDCBAir.wcnf" to 117355113043uL,
        "f49-DC_TotalLoss.seq-B-3-combined-irabcde.wcnf" to 87177360578uL,
        "facebook1.dimacs.wcnf" to 45581uL,
        "github.dimacs.wcnf" to 187405uL,
        "graphstate_6_6_rigetti-agave_8.wcnf" to 6uL,
        "grover-noancilla_4_52_rigetti-agave_8.wcnf" to 42uL,
        "grover-v-chain_4_52_ibmq-casablanca_7.wcnf" to 27uL,
        "guardian.dimacs.wcnf" to 160777uL,
        "inst2.lp.sm-extracted.wcnf" to 97uL,
        "inst10.lp.sm-extracted.wcnf" to 105uL,
        "inst22.lp.sm-extracted.wcnf" to 180uL,
        "instance1.wcnf" to 607uL,
        "instance2.wcnf" to 828uL,
        "ItalyInstance1.xml.wcnf" to 12uL,
        "k50-18-30.rna.pre.wcnf" to 462uL,
        "k50-21-38.rna.pre.wcnf" to 497uL,
        "k100-14-38.rna.pre.wcnf" to 1953uL,
        "k100-20-63.rna.pre.wcnf" to 2030uL,
        "k100-38-60.rna.pre.wcnf" to 1878uL,
        "k100-40-52.rna.pre.wcnf" to 1861uL,
        "k100-73-76.rna.pre.wcnf" to 2008uL,
        "k100-78-85.rna.pre.wcnf" to 1744uL,
        "lisbon-wedding-1-18.wcnf" to 961uL,
        "lisbon-wedding-2-18.wcnf" to 1137uL,
        "lisbon-wedding-3-17.wcnf" to 1035uL,
        "lisbon-wedding-4-18.wcnf" to 803uL,
        "lisbon-wedding-5-17.wcnf" to 802uL,
        "lisbon-wedding-9-17.wcnf" to 394uL,
        "lisbon-wedding-10-17.wcnf" to 377uL,
        "log.8.wcsp.log.wcnf" to 2uL,
        "log.28.wcsp.log.wcnf" to 270105uL,
        "log.408.wcsp.log.wcnf" to 6228uL,
        "log.505.wcsp.log.wcnf" to 21253uL,
        "log.1401.wcsp.log.wcnf" to 459106uL,
        "londonist.dimacs.wcnf" to 70703uL,
        "metro_8_8_5_20_10_6_500_1_0.lp.sm-extracted.wcnf" to 82uL,
        "metro_8_8_5_20_10_6_500_1_7.lp.sm-extracted.wcnf" to 89uL,
        "metro_8_8_5_20_10_6_500_1_9.lp.sm-extracted.wcnf" to 105uL,
        "metro_9_8_7_22_10_6_500_1_1.lp.sm-extracted.wcnf" to 52uL,
        "metro_9_8_7_22_10_6_500_1_2.lp.sm-extracted.wcnf" to 60uL,
        "metro_9_8_7_22_10_6_500_1_3.lp.sm-extracted.wcnf" to 44uL,
        "metro_9_8_7_30_10_6_500_1_5.lp.sm-extracted.wcnf" to 47uL,
        "metro_9_8_7_30_10_6_500_1_6.lp.sm-extracted.wcnf" to 31uL,
        "metro_9_8_7_30_10_6_500_1_7.lp.sm-extracted.wcnf" to 47uL,
        "metro_9_8_7_30_10_6_500_1_8.lp.sm-extracted.wcnf" to 55uL,
        "metro_9_9_10_35_13_7_500_2_7.lp.sm-extracted.wcnf" to 37uL,
        "MinWidthCB_milan_100_12_1k_1s_2t_3.wcnf" to 109520uL,
        "MinWidthCB_milan_200_12_1k_4s_1t_4.wcnf" to 108863uL,
        "MinWidthCB_mitdbsample_100_43_1k_2s_2t_2.wcnf" to 38570uL,
        "MinWidthCB_mitdbsample_100_64_1k_2s_1t_2.wcnf" to 66045uL,
        "MinWidthCB_mitdbsample_200_43_1k_2s_2t_2.wcnf" to 50615uL,
        "MinWidthCB_mitdbsample_200_64_1k_2s_1t_2.wcnf" to 78400uL,
        "MinWidthCB_mitdbsample_200_64_1k_2s_3t_2.wcnf" to 73730uL,
        "MinWidthCB_mitdbsample_300_26_1k_3s_2t_3.wcnf" to 32420uL,
        "MLI.ilpd_train_0_DNF_5_5.wcnf" to 700uL,
        "mul.role_smallcomp_multiple_0.3_6.wcnf" to 139251uL,
        "mul.role_smallcomp_multiple_1.0_6.wcnf" to 295598uL,
        "openstreetmap.dimacs.wcnf" to 65915uL,
        "pac.80cfe9a6-9b1b-11df-965e-00163e46d37a_l1.wcnf" to 1924238uL,
        "pac.fa3d0fb2-db9e-11df-a0ec-00163e3d3b7c_l1.wcnf" to 4569599uL,
        "pac.rand179_l1.wcnf" to 493118uL,
        "pac.rand892_l1.wcnf" to 224702uL,
        "pac.rand984_l1.wcnf" to 345082uL,
        "ped2.B.recomb1-0.01-2.wcnf" to 7uL,
        "ped2.B.recomb1-0.10-7.wcnf" to 588uL,
        "ped3.D.recomb10-0.20-12.wcnf" to 349uL,
        "ped3.D.recomb10-0.20-14.wcnf" to 7uL,
        "portfoliovqe_4_18_rigetti-agave_8.wcnf" to 33uL,
        "power-distribution_1_2.wcnf" to 3uL,
        "power-distribution_1_4.wcnf" to 3uL,
        "power-distribution_1_6.wcnf" to 3uL,
        "power-distribution_1_8.wcnf" to 3uL,
        "power-distribution_2_2.wcnf" to 10uL,
        "power-distribution_2_8.wcnf" to 10uL,
        "power-distribution_3_4.wcnf" to 1uL,
        "power-distribution_7_6.wcnf" to 18uL,
        "power-distribution_8_4.wcnf" to 40uL,
        "power-distribution_8_7.wcnf" to 40uL,
        "power-distribution_9_2.wcnf" to 18uL,
        "power-distribution_11_6.wcnf" to 126uL,
        "power-distribution_12_2.wcnf" to 216uL,
        "power-distribution_12_5.wcnf" to 216uL,
        "qaoa_4_16_ibmq-casablanca_7.wcnf" to 12uL,
        "qft_5_26_ibmq-casablanca_7.wcnf" to 15uL,
        "qftentangled_4_21_ibmq-casablanca_7.wcnf" to 15uL,
        "qftentangled_4_39_rigetti-agave_8.wcnf" to 18uL,
        "qftentangled_5_30_ibmq-london_5.wcnf" to 27uL,
        "qftentangled_5_48_rigetti-agave_8.wcnf" to 24uL,
        "qgan_6_15_ibmq-casablanca_7.wcnf" to 24uL,
        "qpeexact_5_26_ibmq-casablanca_7.wcnf" to 15uL,
        "qwalk-v-chain_3_30_ibmq-casablanca_7.wcnf" to 30uL,
        "qwalk-v-chain_5_102_ibmq-london_5.wcnf" to 81uL,
        "rail507.wcnf" to 174uL,
        "rail516.wcnf" to 182uL,
        "rail582.wcnf" to 211uL,
        "ran.max_cut_60_420_2.asc.wcnf" to 703uL,
        "ran.max_cut_60_420_5.asc.wcnf" to 715uL,
        "ran.max_cut_60_420_9.asc.wcnf" to 674uL,
        "ran.max_cut_60_500_2.asc.wcnf" to 900uL,
        "ran.max_cut_60_560_3.asc.wcnf" to 1054uL,
        "ran.max_cut_60_560_7.asc.wcnf" to 1053uL,
        "ran.max_cut_60_600_1.asc.wcnf" to 1156uL,
        "ran.max_cut_60_600_9.asc.wcnf" to 1149uL,
        "random-dif-2.rna.pre.wcnf" to 929uL,
        "random-dif-9.rna.pre.wcnf" to 456uL,
        "random-dif-16.rna.pre.wcnf" to 768uL,
        "random-dif-25.rna.pre.wcnf" to 512uL,
        "random-net-20-5_network-4.net.wcnf" to 19602uL,
        "random-net-30-3_network-2.net.wcnf" to 27606uL,
        "random-net-30-4_network-3.net.wcnf" to 24925uL,
        "random-net-40-2_network-8.net.wcnf" to 38289uL,
        "random-net-40-2_network-9.net.wcnf" to 35951uL,
        "random-net-40-3_network-5.net.wcnf" to 35488uL,
        "random-net-40-4_network-2.net.wcnf" to 36427uL,
        "random-net-50-3_network-5.net.wcnf" to 41356uL,
        "random-net-50-4_network-8.net.wcnf" to 43243uL,
        "random-net-60-3_network-3.net" to 50929uL,
        "random-net-100-1_network-3.net.wcnf" to 91570uL,
        "random-net-120-1_network-5.net.wcnf" to 117198uL,
        "random-net-220-1_network-7.net.wcnf" to 203783uL,
        "random-net-240-1_network-7.net.wcnf" to 219252uL,
        "random-net-260-1_network-4.net.wcnf" to 238131uL,
        "random-same-5.rna.pre.wcnf" to 456uL,
        "random-same-12.rna.pre.wcnf" to 597uL,
        "random-same-19.rna.pre.wcnf" to 337uL,
        "random-same-25.rna.pre.wcnf" to 224uL,
        "ran-scp.scp41_weighted.wcnf" to 429uL,
        "ran-scp.scp48_weighted.wcnf" to 492uL,
        "ran-scp.scp49_weighted.wcnf" to 641uL,
        "ran-scp.scp51_weighted.wcnf" to 253uL,
        "ran-scp.scp54_weighted.wcnf" to 242uL,
        "ran-scp.scp56_weighted.wcnf" to 213uL,
        "ran-scp.scp58_weighted.wcnf" to 288uL,
        "ran-scp.scp65_weighted.wcnf" to 161uL,
        "ran-scp.scp410_weighted.wcnf" to 514uL,
        "ran-scp.scpnre5_weighted.wcnf" to 28uL,
        "ran-scp.scpnrf1_weighted.wcnf" to 14uL,
        "ran-scp.scpnrf4_weighted.wcnf" to 14uL,
        "ran-scp.scpnrf5_weighted.wcnf" to 13uL,
        "realamprandom_4_72_rigetti-agave_8.wcnf" to 36uL,
        "role_smallcomp_0.7_11.wcnf" to 333834uL,
        "role_smallcomp_0.75_8.wcnf" to 348219uL,
        "role_smallcomp_0.85_4.wcnf" to 369639uL,
        "role_smallcomp_0.85_7.wcnf" to 369639uL,
        "Rounded_BTWBNSL_asia_100_1_3.scores_TWBound_2.wcnf" to 24564427uL,
        "Rounded_BTWBNSL_asia_100_1_3.scores_TWBound_3.wcnf" to 24564427uL,
        "Rounded_BTWBNSL_asia_10000_1_3.scores_TWBound_2.wcnf" to 2247208255uL,
        "Rounded_BTWBNSL_hailfinder_100_1_3.scores_TWBound_2.wcnf" to 602126938uL,
        "Rounded_BTWBNSL_hailfinder_100_1_3.scores_TWBound_3.wcnf" to 601946991uL,
        "Rounded_BTWBNSL_Heart.BIC_TWBound_2.wcnf" to 239742296uL,
        "Rounded_BTWBNSL_insurance_100_1_3.scores_TWBound_2.wcnf" to 170760179uL,
        "Rounded_BTWBNSL_insurance_1000_1_3.scores_TWBound_2.wcnf" to 1389279780uL,
        "Rounded_BTWBNSL_insurance_1000_1_3.scores_TWBound_3.wcnf" to 1388734978uL,
        "Rounded_BTWBNSL_insurance_1000_1_3.scores_TWBound_4.wcnf" to 1388734978uL,
        "Rounded_BTWBNSL_Water_1000_1_2.scores_TWBound_4.wcnf" to 1326306453uL,
        "Rounded_CorrelationClustering_Ionosphere_BINARY_N200_D0.200.wcnf" to 4604640uL,
        "Rounded_CorrelationClustering_Orl_BINARY_N320_D0.200.wcnf" to 4429109uL,
        "Rounded_CorrelationClustering_Protein1_BINARY_N360.wcnf" to 27536228uL,
        "Rounded_CorrelationClustering_Protein2_BINARY_N220.wcnf" to 13727551uL,
        "Rounded_CorrelationClustering_Protein2_UNARY_N100.wcnf" to 3913145uL,
        "simNo_1-s_15-m_50-n_50-fp_0.0001-fn_0.20.wcnf" to 11501657324586uL,
        "simNo_2-s_5-m_100-n_100-fp_0.0001-fn_0.05.wcnf" to 99635408482313uL,
        "simNo_3-s_5-m_50-n_50-fp_0.0001-fn_0.05.wcnf" to 18938961942919uL,
        "simNo_5-s_15-m_100-n_100-fp_0.0001-fn_0.20.wcnf" to 113321765415159uL,
        "simNo_6-s_5-m_100-n_50-fp_0.01-fn_0.05.wcnf" to 90981027155327uL,
        "simNo_6-s_15-m_100-n_50-fp_0.01-fn_0.05.wcnf" to 60142712649443uL,
        "simNo_8-s_5-m_100-n_100-fp_0.0001-fn_0.05.wcnf" to 74156301822200uL,
        "simNo_8-s_5-m_100-n_100-fp_0.0001-fn_0.20.wcnf" to 131749300472480uL,
        "simNo_9-s_5-m_100-n_100-fp_0.0001-fn_0.05.wcnf" to 131749300472480uL,
        "simNo_10-s_15-m_100-n_50-fp_0.01-fn_0.20.wcnf" to 84803002848794uL,
        "simNo_10-s_15-m_100-n_100-fp_0.0001-fn_0.20.wcnf" to 82981983123459uL,
        "su2random_4_18_ibmq-casablanca_7.wcnf" to 24uL,
        "su2random_5_30_ibmq-london_5.wcnf" to 51uL,
        "tcp_students_91_it_2.wcnf" to 3024uL,
        "tcp_students_91_it_3.wcnf" to 2430uL,
        "tcp_students_91_it_6.wcnf" to 2877uL,
        "tcp_students_91_it_7.wcnf" to 2505uL,
        "tcp_students_91_it_13.wcnf" to 2730uL,
        "tcp_students_98_it_8.wcnf" to 2727uL,
        "tcp_students_98_it_9.wcnf" to 2469uL,
        "tcp_students_98_it_12.wcnf" to 2994uL,
        "tcp_students_105_it_7.wcnf" to 3024uL,
        "tcp_students_105_it_13.wcnf" to 3360uL,
        "tcp_students_105_it_15.wcnf" to 3258uL,
        "tcp_students_112_it_1.wcnf" to 3513uL,
        "tcp_students_112_it_3.wcnf" to 2916uL,
        "tcp_students_112_it_5.wcnf" to 3366uL,
        "tcp_students_112_it_7.wcnf" to 3513uL,
        "tcp_students_112_it_15.wcnf" to 3585uL,
        "test1--n-5000.wcnf" to 20uL,
        "test2.wcnf" to 16uL,
        "test2--n-5000.wcnf" to 3uL,
        "test5--n-5000.wcnf" to 2uL,
        "test9--n-5000.wcnf" to 2uL,
        "test18--n-5000.wcnf" to 22uL,
        "test25--n-10000.wcnf" to 4uL,
        "test34--n-10000.wcnf" to 3uL,
        "test41--n-15000.wcnf" to 5uL,
        "test42--n-15000.wcnf" to 2uL,
        "test53--n-15000.wcnf" to 10uL,
        "test54--n-15000.wcnf" to 45uL,
        "test66--n-20000.wcnf" to 1uL,
        "test67--n-20000.wcnf" to 1uL,
        "test70--n-20000.wcnf" to 5uL,
        "test75--n-20000.wcnf" to 5uL,
        "up-.mancoosi-test-i10d0u98-11.wcnf" to 1780771uL,
        "up-.mancoosi-test-i10d0u98-16.wcnf" to 1780806uL,
        "up-.mancoosi-test-i20d0u98-9.wcnf" to 1780788uL,
        "up-.mancoosi-test-i30d0u98-3.wcnf" to 1780860uL,
        "up-.mancoosi-test-i40d0u98-7.wcnf" to 1780807uL,
        "up-.mancoosi-test-i40d0u98-17.wcnf" to 1780852uL,
        "vio.role_smallcomp_violations_0.3_3.wcnf" to 185080uL,
        "vio.role_smallcomp_violations_0.45_8.wcnf" to 244141uL,
        "vqe_4_12_ibmq-casablanca_7.wcnf" to 15uL,
        "vqe_5_20_ibmq-london_5.wcnf" to 33uL,
        "warehouse0.wcsp.wcnf" to 328uL,
        "warehouse1.wcsp.wcnf" to 730567uL,
        "wcn.adult_train_3_DNF_1_5.wcnf" to 24254uL,
        "wcn.ilpd_test_8_CNF_4_20.wcnf" to 287uL,
        "wcn.ionosphere_train_5_DNF_2_10.wcnf" to 47uL,
        "wcn.parkinsons_test_5_CNF_2_10.wcnf" to 58uL,
        "wcn.pima_test_3_CNF_1_5.wcnf" to 125uL,
        "wcn.tictactoe_test_8_CNF_2_20.wcnf" to 346uL,
        "wcn.titanic_test_7_CNF_5_20.wcnf" to 557uL,
        "wcn.titanic_test_8_DNF_1_20.wcnf" to 449uL,
        "wcn.titanic_train_7_CNF_5_15.wcnf" to 3262uL,
        "wcn.titanic_train_8_CNF_5_10.wcnf" to 2201uL,
        "wcn.transfusion_test_7_DNF_3_5.wcnf" to 96uL,
        "wcn.transfusion_train_2_CNF_5_10.wcnf" to 1600uL,
        "wcn.transfusion_train_3_CNF_3_15.wcnf" to 2400uL,
        "WCNF_pathways_p01.wcnf" to 2uL,
        "WCNF_pathways_p03.wcnf" to 30uL,
        "WCNF_pathways_p05.wcnf" to 60uL,
        "WCNF_pathways_p06.wcnf" to 64uL,
        "WCNF_pathways_p08.wcnf" to 182uL,
        "WCNF_pathways_p09.wcnf" to 157uL,
        "WCNF_pathways_p10.wcnf" to 129uL,
        "WCNF_pathways_p12.wcnf" to 188uL,
        "WCNF_pathways_p14.wcnf" to 207uL,
        "WCNF_pathways_p16.wcnf" to 257uL,
        "WCNF_storage_p02.wcnf" to 5uL,
        "WCNF_storage_p06.wcnf" to 173uL,
        "wei.SingleDay_3_weighted.wcnf" to 35439uL,
        "wei.Subnetwork_7_weighted.wcnf" to 43213uL,
        "wei.Subnetwork_9_weighted.wcnf" to 82813uL,
        "wikipedia.dimacs.wcnf" to 42676uL,
        "wpm.mancoosi-test-i1000d0u98-15.wcnf" to 92031744uL,
        "wpm.mancoosi-test-i2000d0u98-25.wcnf" to 332548069uL,
        "wpm.mancoosi-test-i3000d0u98-50.wcnf" to 422725765uL,
        "wpm.mancoosi-test-i3000d0u98-70.wcnf" to 512958012uL,
        "wpm.mancoosi-test-i4000d0u98-76.wcnf" to 738411504uL,
        "youtube.dimacs.wcnf" to 227167uL,
    )
}
