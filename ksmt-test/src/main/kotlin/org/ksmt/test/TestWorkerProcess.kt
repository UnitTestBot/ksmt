package org.ksmt.test

import com.jetbrains.rd.framework.IProtocol
import com.microsoft.z3.AST
import com.microsoft.z3.BoolSort
import com.microsoft.z3.Context
import com.microsoft.z3.Expr
import com.microsoft.z3.Native
import com.microsoft.z3.Solver
import com.microsoft.z3.Status
import org.ksmt.solver.fixtures.yices.KTestYicesContext
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.runner.core.ChildProcessBase
import org.ksmt.runner.core.KsmtWorkerArgs
import org.ksmt.runner.generated.models.TestCheckResult
import org.ksmt.runner.generated.models.TestConversionResult
import org.ksmt.runner.generated.models.TestProtocolModel
import org.ksmt.runner.generated.models.testProtocolModel
import org.ksmt.runner.serializer.AstSerializationCtx
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.KBitwuzlaContext
import org.ksmt.solver.bitwuzla.KBitwuzlaExprConverter
import org.ksmt.solver.bitwuzla.KBitwuzlaExprInternalizer
import org.ksmt.solver.z3.KZ3Context
import org.ksmt.solver.yices.*
import org.ksmt.solver.z3.KZ3ExprConverter
import org.ksmt.solver.z3.KZ3ExprInternalizer
import org.ksmt.solver.z3.KZ3Solver
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.uncheckedCast

class TestWorkerProcess : ChildProcessBase<TestProtocolModel>() {
    private var workerCtx: KContext? = null
    private var workerZ3Ctx: Context? = null

    private val ctx: KContext
        get() = workerCtx ?: error("Solver is not initialized")

    private val z3Ctx: Context
        get() = workerZ3Ctx ?: error("Solver is not initialized")

    private val solvers = mutableListOf<Solver>()
    private val nativeAsts = mutableListOf<AST>()
    private val equalityChecks = mutableMapOf<Int, MutableList<EqualityCheck>>()
    private val equalityCheckAssumptions = mutableMapOf<Int, MutableList<Expr<BoolSort>>>()

    private fun create() {
        workerCtx = KContext()
        workerZ3Ctx = Context()
        equalityChecks.clear()
        equalityCheckAssumptions.clear()
        solvers.clear()
        nativeAsts.clear()
    }

    private fun delete() {
        equalityChecks.clear()
        equalityCheckAssumptions.clear()
        nativeAsts.clear()
        solvers.clear()
        ctx.close()
        z3Ctx.close()
        workerCtx = null
        workerZ3Ctx = null
    }

    private fun parseFile(path: String): List<Long> = try {
        val parsed = z3Ctx.parseSMTLIB2File(
            path,
            emptyArray(),
            emptyArray(),
            emptyArray(),
            emptyArray()
        )
        nativeAsts.addAll(parsed)
        nativeAsts.map { z3Ctx.unwrapAST(it) }
    } catch (ex: Exception) {
        throw SmtLibParseError(ex)
    }

    private fun convertAssertions(nativeAssertions: List<Long>): List<KExpr<KBoolSort>> {
        val converter = KZ3ExprConverter(ctx, KZ3Context(z3Ctx))
        return with(converter) { nativeAssertions.map { it.convertExpr() } }
    }

    private fun internalizeAndConvertBitwuzla(assertions: List<KExpr<KBoolSort>>): List<KExpr<KBoolSort>> =
        KBitwuzlaContext().use { bitwuzlaCtx ->
            val internalizer = KBitwuzlaExprInternalizer(bitwuzlaCtx)
            val bitwuzlaAssertions = with(internalizer) {
                assertions.map { it.internalize() }
            }

            val converter = KBitwuzlaExprConverter(ctx, bitwuzlaCtx)
            with(converter) {
                bitwuzlaAssertions.map { it.convertExpr(ctx.boolSort) }
            }
        }

    private fun internalizeAndConvertYices(assertions: List<KExpr<KBoolSort>>): List<KExpr<KBoolSort>> {
        KTestYicesContext().use { internContext ->
            val internalizer = KYicesExprInternalizer(ctx, internContext)

            val yicesAssertions = with(internalizer) {
                assertions.map { it.internalize() }
            }

            val converter = KYicesExprConverter(ctx, internContext)

            return with(converter) {
                yicesAssertions.map { it.convert() }
            }
        }
    }

    private fun createSolver(timeout: Int): Int {
        val solver = with(z3Ctx) {
            mkSolver().apply {
                val params = mkParams().apply {
                    add("timeout",  timeout)
                    add("random_seed", SEED_FOR_RANDOM)
                }
                setParameters(params)
            }
        }
        solvers.add(solver)
        return solvers.lastIndex
    }

    private fun assert(solver: Int, expr: Long) {
        @Suppress("UNCHECKED_CAST")
        solvers[solver].add(z3Ctx.wrapAST(expr) as Expr<BoolSort>)
    }

    private fun check(solver: Int): KSolverStatus {
        return solvers[solver].check().processCheckResult()
    }

    private fun exprToString(expr: Long): String {
        return z3Ctx.wrapAST(expr).toString()
    }

    private fun getReasonUnknown(solver: Int): String {
        return solvers[solver].reasonUnknown
    }

    private fun addEqualityCheck(solver: Int, actual: KExpr<*>, expected: Long) {
        val actualExpr = internalize(actual)
        val expectedExpr = z3Ctx.wrapAST(expected) as Expr<*>
        val checks = equalityChecks.getOrPut(solver) { mutableListOf() }
        checks += EqualityCheck(actual = actualExpr, expected = expectedExpr)
    }

    private fun addEqualityCheckAssumption(solver: Int, assumption: KExpr<*>) {
        val assumptionExpr = internalize(assumption)
        val assumptions = equalityCheckAssumptions.getOrPut(solver) { mutableListOf() }
        assumptions.add(assumptionExpr.uncheckedCast())
    }

    private fun checkEqualities(solver: Int): KSolverStatus = with(z3Ctx) {
        val checks = equalityChecks[solver] ?: emptyList()
        val equalityBindings = checks.map { mkNot(mkEq(it.actual, it.expected)) }
        equalityCheckAssumptions[solver]?.forEach { solvers[solver].add(it) }
        solvers[solver].add(mkOr(*equalityBindings.toTypedArray()))
        return check(solver)
    }

    private fun findFirstFailedEquality(solver: Int): Int? = with(z3Ctx){
        val solverInstance = solvers[solver]
        equalityCheckAssumptions[solver]?.forEach { solvers[solver].add(it) }
        val checks = equalityChecks[solver] ?: emptyList()
        for ((idx, check) in checks.withIndex()) {
            solverInstance.push()
            val binding = mkNot(mkEq(check.actual, check.expected))
            solverInstance.add(binding)
            val status = solverInstance.check()
            solverInstance.pop()
            if (status == Status.SATISFIABLE)
                return idx
        }
        return null
    }

    private fun mkTrueExpr(): Long = with(z3Ctx) {
        val trueExpr = mkTrue().also { nativeAsts.add(it) }
        unwrapAST(trueExpr)
    }

    private fun Status?.processCheckResult() = when (this) {
        Status.SATISFIABLE -> KSolverStatus.SAT
        Status.UNSATISFIABLE -> KSolverStatus.UNSAT
        Status.UNKNOWN -> KSolverStatus.UNKNOWN
        null -> KSolverStatus.UNKNOWN
    }

    private fun internalize(expr: KExpr<*>): Expr<*> {
        val internalizer = KZ3ExprInternalizer(ctx, KZ3Context(z3Ctx))
        return with(internalizer) { expr.internalizeExprWrapped() }
    }

    private data class EqualityCheck(val actual: Expr<*>, val expected: Expr<*>)

    override fun parseArgs(args: Array<String>) = KsmtWorkerArgs.fromList(args.toList())

    override fun initProtocolModel(protocol: IProtocol): TestProtocolModel =
        protocol.testProtocolModel

    @Suppress("LongMethod")
    override fun TestProtocolModel.setup(astSerializationCtx: AstSerializationCtx) {
        // Limit z3 native memory usage to avoid OOM
        Native.globalParamSet("memory_high_watermark_mb", "2048") // 2048 megabytes

        /**
         *  Memory usage hard limit.
         *
         *  Normally z3 will throw an exception when used
         *  memory amount is slightly above `memory_high_watermark`.
         *  But `memory_high_watermark` check may be missed somewhere in Z3 and
         *  memory usage will become a way higher than limit.
         *  Therefore, we use hard limit to avoid OOM
         */
        Native.globalParamSet("memory_max_size", "4096") // 4096 megabytes

        create.measureExecutionForTermination {
            create()
            astSerializationCtx.initCtx(ctx)
        }
        delete.measureExecutionForTermination {
            astSerializationCtx.resetCtx()
            delete()
        }
        parseFile.measureExecutionForTermination { path ->
            parseFile(path)
        }
        convertAssertions.measureExecutionForTermination { assertions ->
            val converted = convertAssertions(assertions)
            TestConversionResult(converted)
        }
        internalizeAndConvertBitwuzla.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            val converted = internalizeAndConvertBitwuzla(params.expressions as List<KExpr<KBoolSort>>)
            TestConversionResult(converted)
        }
        internalizeAndConvertYices.measureExecutionForTermination { params ->
            @Suppress("UNCHECKED_CAST")
            val converted = internalizeAndConvertYices(params.expressions as List<KExpr<KBoolSort>>)
            TestConversionResult(converted)
        }
        createSolver.measureExecutionForTermination { timeout ->
            createSolver(timeout)
        }
        assert.measureExecutionForTermination { params ->
            assert(params.solver, params.expr)
        }
        check.measureExecutionForTermination { solver ->
            val status = check(solver)
            TestCheckResult(status)
        }
        exprToString.measureExecutionForTermination { solver ->
            exprToString(solver)
        }
        getReasonUnknown.measureExecutionForTermination { solver ->
            getReasonUnknown(solver)
        }
        addEqualityCheck.measureExecutionForTermination { params ->
            addEqualityCheck(params.solver, params.actual as KExpr<*>, params.expected)
        }
        addEqualityCheckAssumption.measureExecutionForTermination { params ->
            addEqualityCheckAssumption(params.solver, params.assumption as KExpr<*>)
        }
        checkEqualities.measureExecutionForTermination { solver ->
            val status = checkEqualities(solver)
            TestCheckResult(status)
        }
        findFirstFailedEquality.measureExecutionForTermination { solver ->
            findFirstFailedEquality(solver)
        }
        mkTrueExpr.measureExecutionForTermination {
            mkTrueExpr()
        }
    }

    companion object {
        private const val SEED_FOR_RANDOM = 12345

        init {
            // force native library load
            KZ3Solver(KContext()).close()
        }

        @JvmStatic
        fun main(args: Array<String>) {
            TestWorkerProcess().start(args)
        }
    }
}
