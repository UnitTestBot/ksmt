package io.ksmt.test

import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.expr.KFpRoundingMode.RoundNearestTiesToEven
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.bitwuzla.KBitwuzlaSolver
import io.ksmt.solver.cvc5.KCvc5Solver
import io.ksmt.solver.yices.KYicesSolver
import io.ksmt.solver.z3.KZ3Solver
import io.ksmt.sort.KBoolSort
import io.ksmt.sort.KSort
import io.ksmt.symfpu.solver.KSymFpuSolver
import io.ksmt.test.ModelGenerationTest.Companion.FOp
import io.ksmt.test.ModelGenerationTest.Companion.Op
import org.junit.jupiter.api.AfterEach
import org.junit.jupiter.api.Assumptions
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.params.ParameterizedTest
import org.junit.jupiter.params.provider.Arguments
import org.junit.jupiter.params.provider.MethodSource
import kotlin.test.assertEquals
import kotlin.time.Duration.Companion.seconds

class Z3ModelGenerationTest : ModelGenerationTest() {
    override fun mkSolver(ctx: KContext): KSolver<*> = KZ3Solver(ctx)

    @ParameterizedTest
    @MethodSource("basicOperationsArgs")
    fun basicOperations(op: Op<KSort>) = testBasicOp(op)

    @ParameterizedTest
    @MethodSource("functionalOperationsArgs")
    fun functionalOperations(op: FOp<KSort>) = testFunctionalOp(op)
}

class BitwuzlaModelGenerationTest : ModelGenerationTest() {
    override fun mkSolver(ctx: KContext): KSolver<*> = KBitwuzlaSolver(ctx)

    @ParameterizedTest
    @MethodSource("basicOperationsArgs")
    fun basicOperations(op: Op<KSort>) = testBasicOp(op)

    @ParameterizedTest
    @MethodSource("functionalOperationsArgs")
    fun functionalOperations(op: FOp<KSort>) = testFunctionalOp(op)
}

class YicesModelGenerationTest : ModelGenerationTest() {
    override fun mkSolver(ctx: KContext): KSolver<*> = KYicesSolver(ctx)

    @ParameterizedTest
    @MethodSource("basicOperationsArgs")
    fun basicOperations(op: Op<KSort>) = testBasicOp(op)

    @ParameterizedTest
    @MethodSource("functionalOperationsArgs")
    fun functionalOperations(op: FOp<KSort>) = testFunctionalOp(op)
}

class CvcModelGenerationTest : ModelGenerationTest() {
    override fun mkSolver(ctx: KContext): KSolver<*> = KCvc5Solver(ctx)

    @ParameterizedTest
    @MethodSource("basicOperationsArgs")
    fun basicOperations(op: Op<KSort>) = testBasicOp(op)

    @ParameterizedTest
    @MethodSource("functionalOperationsArgs")
    fun functionalOperations(op: FOp<KSort>) = testFunctionalOp(op)
}

class YicesWithSymFpuModelGenerationTest : ModelGenerationTest() {
    override fun mkSolver(ctx: KContext): KSolver<*> = KSymFpuSolver(KYicesSolver(ctx), ctx)

    @ParameterizedTest
    @MethodSource("basicOperationsArgs")
    fun basicOperations(op: Op<KSort>) = testBasicOp(op)

    @ParameterizedTest
    @MethodSource("functionalOperationsArgs")
    fun functionalOperations(op: FOp<KSort>) = testFunctionalOp(op)
}

abstract class ModelGenerationTest {
    lateinit var context: KContext
    lateinit var solver: KSolver<*>

    abstract fun mkSolver(ctx: KContext): KSolver<*>

    @BeforeEach
    fun initSolver() {
        context = KContext()
        solver = mkSolver(context)
    }

    @AfterEach
    fun restSolver() {
        solver.close()
        context.close()
    }

    fun testBasicOp(op: Op<KSort>): Unit = with(context) {
        val vf = VarFactory(context)
        val expr = op.mkOp(context, vf)
        val res = vf.v(expr.sort)

        assertOp(op, res eq expr)

        repeat(5) {
            val model = checkOp(op) ?: return

            val modelValue = model.eval(expr)

            val detachedModel = model.detach()
            val detachedValue = detachedModel.eval(expr)

            if (modelValue != detachedValue) {
                if (op.unspecified) {
                    System.err.println("POSSIBLY ERROR: (UNSPECIFIED) $op | SOLVER=$modelValue KSMT=$detachedValue")
                } else {
                    assertEquals(modelValue, detachedValue)
                }
            }

            assertOp(op, res neq modelValue)
        }
    }

    fun testFunctionalOp(op: FOp<KSort>) {
        val vf = VarFactory(context)
        val expr = op.mkOp(context, vf)
        val checkExpr = op.checkOp(context, vf, expr)

        assertOp(op, checkExpr)

        val model = checkOp(op) ?: return

        val modelValue = model.eval(expr)

        val detachedModel = model.detach()
        val detachedValue = detachedModel.eval(expr)

        assertEquals(modelValue, detachedValue)
    }

    private fun assertOp(op: Op<*>, expr: KExpr<KBoolSort>) = try {
        solver.assert(expr)
    } catch (ex: KSolverException) {
        System.err.println("IGNORE: (UNSUPPORTED) $op | ${ex.message}")
        Assumptions.assumeTrue(false)
    }

    private fun checkOp(op: Op<*>): KModel? {
        val status = solver.check(timeout = 10.seconds)
        return when (status) {
            KSolverStatus.SAT -> solver.model()
            KSolverStatus.UNSAT -> null
            KSolverStatus.UNKNOWN -> {
                System.err.println("IGNORE: (UNKNOWN) $op | ${solver.reasonOfUnknown()}")
                Assumptions.assumeTrue(false)
                null
            }
        }
    }

    companion object {
        class VarFactory(val ctx: KContext) {
            fun <T : KSort> v(sort: T): KExpr<T> = ctx.mkFreshConst("x", sort)
        }

        open class Op<T : KSort>(
            val unspecified: Boolean,
            val mkOp: KContext.(VarFactory) -> KExpr<T>
        ) {
            override fun toString(): String = KContext().use { ctx ->
                val vf = VarFactory(ctx)
                "${mkOp(ctx, vf)}"
            }
        }

        class FOp<T : KSort>(
            unspecified: Boolean,
            mkOp: KContext.(VarFactory) -> KExpr<T>,
            val checkOp: KContext.(VarFactory, KExpr<T>) -> KExpr<KBoolSort>,
        ) : Op<T>(unspecified, mkOp)

        private fun <T : KSort> MutableList<Op<*>>.op(mkOp: KContext.(VarFactory) -> KExpr<T>) {
            this += Op(unspecified = false, mkOp = mkOp)
        }

        data class UnspecifiedOp<T : KSort>(
            val op: KExpr<T>,
            val unspecifiedIf: KExpr<KBoolSort>,
            val unspecifiedValue: KExpr<T>,
        )

        private fun <T : KSort> MutableList<Op<*>>.unspecifiedOp(
            mkOp: KContext.(VarFactory) -> UnspecifiedOp<T>
        ) {
            this += Op(unspecified = true, mkOp = { vf -> mkOp(this, vf).op })
            this += Op(unspecified = false, mkOp = { vf ->
                val op = mkOp(this, vf)
                mkIteNoSimplify(op.unspecifiedIf, op.unspecifiedValue, op.op)
            })
        }

        private fun <T : KSort> MutableList<FOp<*>>.functionalOp(
            mkOp: KContext.(VarFactory) -> KExpr<T>,
            checkOp: KContext.(VarFactory, KExpr<T>) -> KExpr<KBoolSort>,
        ) {
            this += FOp(unspecified = false, mkOp = mkOp, checkOp = checkOp)
        }

        @JvmStatic
        fun basicOperationsArgs() = basicOperations().map { Arguments.of(it) }

        @JvmStatic
        fun functionalOperationsArgs() = functionalOperations().map { Arguments.of(it) }

        @Suppress("LongMethod")
        private fun basicOperations() = buildList {
            op { mkAndNoSimplify(it.v(boolSort), it.v(boolSort)) }
            op { mkAndNoSimplify(listOf(it.v(boolSort), it.v(boolSort), it.v(boolSort))) }
            op { mkOrNoSimplify(it.v(boolSort), it.v(boolSort)) }
            op { mkOrNoSimplify(listOf(it.v(boolSort), it.v(boolSort), it.v(boolSort))) }
            op { mkImpliesNoSimplify(it.v(boolSort), it.v(boolSort)) }
            op { mkXorNoSimplify(it.v(boolSort), it.v(boolSort)) }
            op { mkNotNoSimplify(it.v(boolSort)) }

            op { mkDistinctNoSimplify(listOf(it.v(bv32Sort), it.v(bv32Sort), it.v(bv32Sort))) }
            op { mkEqNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkIteNoSimplify(it.v(boolSort), it.v(bv32Sort), it.v(bv32Sort)) }

            op { mkArithAddNoSimplify(listOf(it.v(intSort), it.v(intSort))) }
            op { mkArithMulNoSimplify(listOf(it.v(intSort), it.v(intSort))) }
            op { mkArithSubNoSimplify(listOf(it.v(intSort), it.v(intSort))) }
            op { mkArithAddNoSimplify(listOf(it.v(realSort), it.v(realSort))) }
            op { mkArithMulNoSimplify(listOf(it.v(realSort), it.v(realSort))) }
            op { mkArithSubNoSimplify(listOf(it.v(realSort), it.v(realSort))) }

            unspecifiedOp {
                val divisor = it.v(intSort)
                UnspecifiedOp(
                    op = mkArithDivNoSimplify(it.v(intSort), divisor),
                    unspecifiedIf = divisor eq 0.expr,
                    unspecifiedValue = 0.expr
                )
            }
            op { mkArithGeNoSimplify(it.v(intSort), it.v(intSort)) }
            op { mkArithGtNoSimplify(it.v(intSort), it.v(intSort)) }
            op { mkArithLeNoSimplify(it.v(intSort), it.v(intSort)) }
            op { mkArithLtNoSimplify(it.v(intSort), it.v(intSort)) }
            unspecifiedOp {
                val base = it.v(intSort)
                val power = it.v(intSort)
                UnspecifiedOp(
                    op = mkArithPowerNoSimplify(base, power),
                    unspecifiedIf = (base eq 0.expr) and (power eq 0.expr),
                    unspecifiedValue = 0.expr,
                )
            }
            op { mkArithUnaryMinusNoSimplify(it.v(intSort)) }

            unspecifiedOp {
                val divisor = it.v(realSort)
                val zero = mkRealNum(0)
                UnspecifiedOp(
                    op = mkArithDivNoSimplify(it.v(realSort), divisor),
                    unspecifiedIf = divisor eq zero,
                    unspecifiedValue = zero
                )
            }
            op { mkArithGeNoSimplify(it.v(realSort), it.v(realSort)) }
            op { mkArithGtNoSimplify(it.v(realSort), it.v(realSort)) }
            op { mkArithLeNoSimplify(it.v(realSort), it.v(realSort)) }
            op { mkArithLtNoSimplify(it.v(realSort), it.v(realSort)) }
            unspecifiedOp {
                val base = it.v(realSort)
                val power = it.v(realSort)
                val zero = mkRealNum(0)
                UnspecifiedOp(
                    op = mkArithPowerNoSimplify(base, power),
                    unspecifiedIf = (base eq zero) and (power eq zero),
                    unspecifiedValue = zero,
                )
            }
            op { mkArithUnaryMinusNoSimplify(it.v(realSort)) }

            unspecifiedOp {
                val divisor = it.v(intSort)
                UnspecifiedOp(
                    op = mkIntModNoSimplify(it.v(intSort), divisor),
                    unspecifiedIf = divisor eq 0.expr,
                    unspecifiedValue = 0.expr
                )
            }
            unspecifiedOp {
                val divisor = it.v(intSort)
                UnspecifiedOp(
                    op = mkIntRemNoSimplify(it.v(intSort), divisor),
                    unspecifiedIf = divisor eq 0.expr,
                    unspecifiedValue = 0.expr
                )
            }
            op { mkIntToRealNoSimplify(it.v(intSort)) }

            op { mkRealIsIntNoSimplify(it.v(realSort)) }
            op { mkRealToIntNoSimplify(it.v(realSort)) }

            op { mkArraySelectNoSimplify(it.v(mkArraySort(bv32Sort, bv32Sort)), it.v(bv32Sort)) }
            op {
                mkArraySelectNoSimplify(
                    it.v(mkArraySort(bv32Sort, bv32Sort, bv32Sort)),
                    it.v(bv32Sort),
                    it.v(bv32Sort)
                )
            }
            op {
                mkArraySelectNoSimplify(
                    it.v(mkArraySort(bv32Sort, bv32Sort, bv32Sort, bv32Sort)),
                    it.v(bv32Sort),
                    it.v(bv32Sort),
                    it.v(bv32Sort)
                )
            }
            op { vf ->
                mkArrayNSelectNoSimplify(
                    vf.v(mkArrayNSort(List(5) { bv32Sort }, bv32Sort)),
                    List(5) { vf.v(bv32Sort) }
                )
            }
            op {
                mkArraySelectNoSimplify(
                    it.v(mkArraySort(mkUninterpretedSort("A"), mkUninterpretedSort("R"))),
                    it.v(mkUninterpretedSort("A"))
                )
            }
            op {
                mkArraySelectNoSimplify(
                    it.v(mkArraySort(mkUninterpretedSort("A"), mkUninterpretedSort("B"), mkUninterpretedSort("R"))),
                    it.v(mkUninterpretedSort("A")),
                    it.v(mkUninterpretedSort("B"))
                )
            }
            op {
                mkArraySelectNoSimplify(
                    it.v(
                        mkArraySort(
                            mkUninterpretedSort("A"),
                            mkUninterpretedSort("B"),
                            mkUninterpretedSort("C"),
                            mkUninterpretedSort("R")
                        )
                    ),
                    it.v(mkUninterpretedSort("A")),
                    it.v(mkUninterpretedSort("B")),
                    it.v(mkUninterpretedSort("C"))
                )
            }
            op { vf ->
                mkArrayNSelectNoSimplify(
                    vf.v(mkArrayNSort(List(5) { mkUninterpretedSort("$it") }, mkUninterpretedSort("R"))),
                    List(5) { vf.v(mkUninterpretedSort("$it")) }
                )
            }

            op { mkBvAddExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvAddNoOverflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort), isSigned = true) }
            op { mkBvAddNoOverflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort), isSigned = false) }
            op { mkBvAddNoUnderflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvAndExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvArithShiftRightExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvConcatExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            unspecifiedOp {
                val divisor = it.v(bv32Sort)
                UnspecifiedOp(
                    op = mkBvDivNoOverflowExprNoSimplify(it.v(bv32Sort), divisor),
                    unspecifiedIf = divisor eq mkBv(0),
                    unspecifiedValue = falseExpr,
                )
            }
            op { mkBvExtractExprNoSimplify(high = 17, low = 5, it.v(bv32Sort)) }
            op { mkBvLogicalShiftRightExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvMulExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvMulNoOverflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort), isSigned = true) }
            op { mkBvMulNoOverflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort), isSigned = false) }
            op { mkBvMulNoUnderflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvNAndExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvNegationExprNoSimplify(it.v(bv32Sort)) }
            op { mkBvNegationNoOverflowExprNoSimplify(it.v(bv32Sort)) }
            op { mkBvNorExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvNotExprNoSimplify(it.v(bv32Sort)) }
            op { mkBvOrExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvReductionAndExprNoSimplify(it.v(bv32Sort)) }
            op { mkBvReductionOrExprNoSimplify(it.v(bv32Sort)) }
            op { mkBvRepeatExprNoSimplify(repeatNumber = 17, it.v(bv32Sort)) }
            op { mkBvRotateLeftExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvRotateLeftIndexedExprNoSimplify(rotation = 17, it.v(bv32Sort)) }
            op { mkBvRotateRightExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvRotateRightIndexedExprNoSimplify(rotation = 17, it.v(bv32Sort)) }
            op { mkBvShiftLeftExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvSignExtensionExprNoSimplify(extensionSize = 17, it.v(bv32Sort)) }
            unspecifiedOp {
                val divisor = it.v(bv32Sort)
                UnspecifiedOp(
                    op = mkBvSignedDivExprNoSimplify(it.v(bv32Sort), divisor),
                    unspecifiedIf = divisor eq mkBv(0),
                    unspecifiedValue = mkBv(0),
                )
            }
            op { mkBvSignedGreaterExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvSignedGreaterOrEqualExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvSignedLessExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvSignedLessOrEqualExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            unspecifiedOp {
                val divisor = it.v(bv32Sort)
                UnspecifiedOp(
                    op = mkBvSignedModExprNoSimplify(it.v(bv32Sort), divisor),
                    unspecifiedIf = divisor eq mkBv(0),
                    unspecifiedValue = mkBv(0),
                )
            }
            unspecifiedOp {
                val divisor = it.v(bv32Sort)
                UnspecifiedOp(
                    op = mkBvSignedRemExprNoSimplify(it.v(bv32Sort), divisor),
                    unspecifiedIf = divisor eq mkBv(0),
                    unspecifiedValue = mkBv(0),
                )
            }
            op { mkBvSubExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvSubNoOverflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvSubNoUnderflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort), isSigned = true) }
            op { mkBvSubNoUnderflowExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort), isSigned = false) }
            unspecifiedOp {
                val divisor = it.v(bv32Sort)
                UnspecifiedOp(
                    op = mkBvUnsignedDivExprNoSimplify(it.v(bv32Sort), divisor),
                    unspecifiedIf = divisor eq mkBv(0),
                    unspecifiedValue = mkBv(0),
                )
            }
            op { mkBvUnsignedGreaterExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvUnsignedGreaterOrEqualExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvUnsignedLessExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvUnsignedLessOrEqualExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            unspecifiedOp {
                val divisor = it.v(bv32Sort)
                UnspecifiedOp(
                    op = mkBvUnsignedRemExprNoSimplify(it.v(bv32Sort), divisor),
                    unspecifiedIf = divisor eq mkBv(0),
                    unspecifiedValue = mkBv(0),
                )
            }
            op { mkBvXNorExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvXorExprNoSimplify(it.v(bv32Sort), it.v(bv32Sort)) }
            op { mkBvZeroExtensionExprNoSimplify(extensionSize = 17, it.v(bv32Sort)) }

            op { mkFpAbsExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpAddExprNoSimplify(it.v(mkFpRoundingModeSort()), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpAddExprNoSimplify(mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpDivExprNoSimplify(it.v(mkFpRoundingModeSort()), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpDivExprNoSimplify(mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpEqualExprNoSimplify(it.v(fp32Sort), it.v(fp32Sort)) }
            op {
                mkFpFusedMulAddExprNoSimplify(
                    it.v(mkFpRoundingModeSort()),
                    it.v(fp32Sort),
                    it.v(fp32Sort),
                    it.v(fp32Sort)
                )
            }
            op {
                mkFpFusedMulAddExprNoSimplify(
                    mkFpRoundingModeExpr(RoundNearestTiesToEven),
                    it.v(fp32Sort),
                    it.v(fp32Sort),
                    it.v(fp32Sort)
                )
            }
            op { mkFpGreaterExprNoSimplify(it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpGreaterOrEqualExprNoSimplify(it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpIsInfiniteExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpIsNaNExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpIsNegativeExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpIsNormalExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpIsPositiveExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpIsSubnormalExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpIsZeroExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpLessExprNoSimplify(it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpLessOrEqualExprNoSimplify(it.v(fp32Sort), it.v(fp32Sort)) }
            unspecifiedOp {
                val lhs = it.v(fp32Sort)
                val rhs = it.v(fp32Sort)
                UnspecifiedOp(
                    op = mkFpMaxExprNoSimplify(lhs, rhs),
                    unspecifiedIf = mkFpIsZeroExpr(lhs) and mkFpIsZeroExpr(rhs) and (mkFpIsPositiveExpr(lhs) neq mkFpIsPositiveExpr(
                        rhs
                    )),
                    unspecifiedValue = mkFpZero(signBit = false, sort = fp32Sort),
                )
            }
            unspecifiedOp {
                val lhs = it.v(fp32Sort)
                val rhs = it.v(fp32Sort)
                UnspecifiedOp(
                    op = mkFpMinExprNoSimplify(lhs, rhs),
                    unspecifiedIf = mkFpIsZeroExpr(lhs) and mkFpIsZeroExpr(rhs) and (mkFpIsPositiveExpr(lhs) neq mkFpIsPositiveExpr(
                        rhs
                    )),
                    unspecifiedValue = mkFpZero(signBit = false, sort = fp32Sort),
                )
            }
            op { mkFpMulExprNoSimplify(it.v(mkFpRoundingModeSort()), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpMulExprNoSimplify(mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpNegationExprNoSimplify(it.v(fp32Sort)) }
            op { mkFpSubExprNoSimplify(it.v(mkFpRoundingModeSort()), it.v(fp32Sort), it.v(fp32Sort)) }
            op { mkFpSubExprNoSimplify(mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp32Sort), it.v(fp32Sort)) }

            op { mkFpRemExprNoSimplify(it.v(fp16Sort), it.v(fp16Sort)) }
            op { mkFpRoundToIntegralExprNoSimplify(it.v(mkFpRoundingModeSort()), it.v(fp16Sort)) }
            op { mkFpRoundToIntegralExprNoSimplify(mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp16Sort)) }
            op { mkFpSqrtExprNoSimplify(it.v(mkFpRoundingModeSort()), it.v(fp16Sort)) }
            op { mkFpSqrtExprNoSimplify(mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp16Sort)) }

            op { mkBv2IntExprNoSimplify(it.v(bv32Sort), isSigned = true) }
            op { mkBv2IntExprNoSimplify(it.v(bv32Sort), isSigned = false) }

            op { mkRealToFpExprNoSimplify(fp32Sort, it.v(mkFpRoundingModeSort()), it.v(realSort)) }
            op { mkRealToFpExprNoSimplify(fp32Sort, mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(realSort)) }
            op { mkBvToFpExprNoSimplify(fp32Sort, it.v(mkFpRoundingModeSort()), it.v(bv32Sort), signed = true) }
            op { mkBvToFpExprNoSimplify(fp32Sort, it.v(mkFpRoundingModeSort()), it.v(bv32Sort), signed = false) }
            op {
                mkBvToFpExprNoSimplify(
                    fp32Sort,
                    mkFpRoundingModeExpr(RoundNearestTiesToEven),
                    it.v(bv32Sort),
                    signed = true
                )
            }
            op {
                mkBvToFpExprNoSimplify(
                    fp32Sort,
                    mkFpRoundingModeExpr(RoundNearestTiesToEven),
                    it.v(bv32Sort),
                    signed = false
                )
            }
            unspecifiedOp {
                val value = it.v(fp16Sort)
                UnspecifiedOp(
                    op = mkFpToBvExprNoSimplify(
                        it.v(mkFpRoundingModeSort()),
                        value,
                        bvSize = bv64Sort.sizeBits.toInt(),
                        isSigned = true
                    ),
                    unspecifiedIf = mkFpIsNaNExpr(value) or mkFpIsInfiniteExpr(value),
                    unspecifiedValue = mkBv(0, bv64Sort),
                )
            }
            unspecifiedOp {
                val value = it.v(fp16Sort)
                UnspecifiedOp(
                    op = mkFpToBvExprNoSimplify(
                        it.v(mkFpRoundingModeSort()),
                        value,
                        bvSize = bv64Sort.sizeBits.toInt(),
                        isSigned = false
                    ),
                    unspecifiedIf = mkFpIsNaNExpr(value) or mkFpIsInfiniteExpr(value) or mkFpIsNegativeExpr(value),
                    unspecifiedValue = mkBv(0, bv64Sort),
                )
            }
            unspecifiedOp {
                val value = it.v(fp16Sort)
                UnspecifiedOp(
                    op = mkFpToBvExprNoSimplify(
                        mkFpRoundingModeExpr(RoundNearestTiesToEven),
                        value,
                        bvSize = bv64Sort.sizeBits.toInt(),
                        isSigned = true
                    ),
                    unspecifiedIf = mkFpIsNaNExpr(value) or mkFpIsInfiniteExpr(value),
                    unspecifiedValue = mkBv(0, bv64Sort),
                )
            }
            unspecifiedOp {
                val value = it.v(fp16Sort)
                UnspecifiedOp(
                    op = mkFpToBvExprNoSimplify(
                        mkFpRoundingModeExpr(RoundNearestTiesToEven),
                        value,
                        bvSize = bv64Sort.sizeBits.toInt(),
                        isSigned = false
                    ),
                    unspecifiedIf = mkFpIsNaNExpr(value) or mkFpIsInfiniteExpr(value) or mkFpIsNegativeExpr(value),
                    unspecifiedValue = mkBv(0, bv64Sort),
                )
            }
            op { mkFpFromBvExprNoSimplify(it.v(bv1Sort), it.v(bv16Sort), it.v(bv32Sort)) }
            op { mkFpToIEEEBvExprNoSimplify(it.v(fp32Sort)) }
            unspecifiedOp {
                val value = it.v(fp16Sort)
                UnspecifiedOp(
                    op = mkFpToRealExprNoSimplify(value),
                    unspecifiedIf = mkFpIsNaNExpr(value) or mkFpIsInfiniteExpr(value),
                    unspecifiedValue = mkRealNum(0),
                )
            }
            op { mkFpToFpExprNoSimplify(fp32Sort, it.v(mkFpRoundingModeSort()), it.v(fp16Sort)) }
            op { mkFpToFpExprNoSimplify(fp32Sort, mkFpRoundingModeExpr(RoundNearestTiesToEven), it.v(fp16Sort)) }
        }

        @Suppress("LongMethod")
        private fun functionalOperations() = buildList {
            functionalOp(
                mkOp = { vf ->
                    mkArrayStoreNoSimplify(
                        vf.v(mkArraySort(bv32Sort, bv32Sort)),
                        vf.v(bv32Sort),
                        vf.v(bv32Sort)
                    )
                },
                checkOp = { vf, expr ->
                    mkArraySelect(expr, vf.v(bv32Sort)) eq vf.v(bv32Sort)
                }
            )
            functionalOp(
                mkOp = { vf ->
                    mkArrayStoreNoSimplify(
                        vf.v(mkArraySort(bv32Sort, bv32Sort, bv32Sort)), vf.v(bv32Sort), vf.v(bv32Sort), vf.v(bv32Sort)
                    )
                },
                checkOp = { vf, expr ->
                    mkArraySelect(expr, vf.v(bv32Sort), vf.v(bv32Sort)) eq vf.v(bv32Sort)
                }
            )
            functionalOp(
                mkOp = { vf ->
                    mkArrayStoreNoSimplify(
                        vf.v(mkArraySort(bv32Sort, bv32Sort, bv32Sort, bv32Sort)),
                        vf.v(bv32Sort),
                        vf.v(bv32Sort),
                        vf.v(bv32Sort),
                        vf.v(bv32Sort)
                    )
                },
                checkOp = { vf, expr ->
                    mkArraySelect(expr, vf.v(bv32Sort), vf.v(bv32Sort), vf.v(bv32Sort)) eq vf.v(bv32Sort)
                }
            )
            functionalOp(
                mkOp = { vf ->
                    mkArrayNStoreNoSimplify(
                        vf.v(mkArrayNSort(List(5) { bv32Sort }, bv32Sort)),
                        List(5) { vf.v(bv32Sort) },
                        vf.v(bv32Sort)
                    )
                },
                checkOp = { vf, expr ->
                    mkArrayNSelect(expr, List(5) { vf.v(bv32Sort) }) eq vf.v(bv32Sort)
                }
            )
            functionalOp(
                mkOp = { vf ->
                    vf.v(mkArraySort(bv32Sort, bv32Sort))
                },
                checkOp = { vf, expr -> expr eq vf.v(expr.sort) }
            )
            functionalOp(
                mkOp = { vf ->
                    vf.v(mkArraySort(bv32Sort, bv32Sort, bv32Sort))
                },
                checkOp = { vf, expr -> expr eq vf.v(expr.sort) }
            )
            functionalOp(
                mkOp = { vf ->
                    vf.v(mkArraySort(bv32Sort, bv32Sort, bv32Sort, bv32Sort))
                },
                checkOp = { vf, expr -> expr eq vf.v(expr.sort) }
            )
            functionalOp(
                mkOp = { vf ->
                    vf.v(mkArrayNSort(List(5) { bv32Sort }, bv32Sort))
                },
                checkOp = { vf, expr -> expr eq vf.v(expr.sort) }
            )
        }
    }
}
