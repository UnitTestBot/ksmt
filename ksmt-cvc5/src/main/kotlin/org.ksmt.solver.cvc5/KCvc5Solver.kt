package org.ksmt.solver.cvc5

import io.github.cvc5.*
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.NativeLibraryLoader
import org.ksmt.utils.mkFreshConst
import java.util.*
import kotlin.time.Duration
import kotlin.time.DurationUnit

open class KCvc5Solver(private val ctx: KContext) : KSolver<KCvc5SolverConfiguration> {
    private val solver = Solver()
    private val cvc5Ctx = KCvc5Context(solver)

    private val exprInternalizer by lazy {
        createExprInternalizer(cvc5Ctx)
    }

    private val exprConverter: KCvc5ExprConverter by lazy {
        KCvc5ExprConverter(ctx, cvc5Ctx)
    }

    private val currentScope: UInt
        get() = cvc5TrackedAssertions.lastIndex.toUInt()

    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var cvc5LastUnknownExplanation: UnknownExplanation? = null

    // we can't use hashset with Term (hashcode is not implemented)
    private var cvc5CurrentLevelTrackedAssertions = TreeSet<Term>()
    private val cvc5TrackedAssertions = mutableListOf(cvc5CurrentLevelTrackedAssertions)

    private var cvc5CurrentLevelAssertions = TreeSet<Term>()
    private val cvc5Assertions = mutableListOf(cvc5CurrentLevelAssertions)


    init {
        solver.setOption("produce-models", "true")
        solver.setOption("produce-unsat-cores", "true")
    }

    open fun createExprInternalizer(cvc5Ctx: KCvc5Context): KCvc5ExprInternalizer = KCvc5ExprInternalizer(ctx, cvc5Ctx)

    override fun configure(configurator: KCvc5SolverConfiguration.() -> Unit) {
        KCvc5SolverConfigurationImpl(solver).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = cvc5Try {
        val cvc5Expr = with(exprInternalizer) { expr.internalizeExpr() }
        cvc5CurrentLevelAssertions.add(cvc5Expr)

        solver.assertFormula(cvc5Expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort> = cvc5Try {
        val trackVar = with(ctx) { boolSort.mkFreshConst("track") }
        val trackedExpr = with(ctx) { trackVar implies expr }

        val cvc5TrackVar = with(exprInternalizer) { trackVar.internalizeExpr() }
        val cvc5TrackedExpr = with(exprInternalizer) { trackedExpr.internalizeExpr() }

        cvc5CurrentLevelTrackedAssertions.add(cvc5TrackVar)
        cvc5CurrentLevelAssertions.add(cvc5TrackedExpr)

        solver.assertFormula(cvc5TrackedExpr)
        solver.assertFormula(cvc5TrackVar)

        trackVar
    }

    override fun push() = solver.push().also {
        cvc5CurrentLevelTrackedAssertions = TreeSet()
        cvc5TrackedAssertions.add(cvc5CurrentLevelTrackedAssertions)

        cvc5CurrentLevelAssertions = TreeSet()
        cvc5Assertions.add(cvc5CurrentLevelAssertions)
    }

    override fun pop(n: UInt) {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }

        if (n == 0u) return

        repeat(n.toInt()) {
            cvc5TrackedAssertions.removeLast()
            cvc5Assertions.removeLast()
        }
        cvc5CurrentLevelTrackedAssertions = cvc5TrackedAssertions.last()
        cvc5CurrentLevelAssertions = cvc5Assertions.last()

        solver.pop(n.toInt())
    }

    override fun check(timeout: Duration): KSolverStatus = cvc5Try {
        solver.updateTimeout(timeout)
        solver.checkSat().processCheckResult()
    }

    override fun reasonOfUnknown(): String = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) { "Unknown reason is only available after UNKNOWN checks" }
        cvc5LastUnknownExplanation?.name ?: "no explanation"
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus = cvc5Try {
        val cvc5Assumptions = with(exprInternalizer) { assumptions.map { it.internalizeExpr() } }.toTypedArray()
        solver.updateTimeout(timeout)
        solver.checkSatAssuming(cvc5Assumptions).processCheckResult()
    }

    override fun model(): KModel = cvc5Try {
        require(lastCheckStatus == KSolverStatus.SAT) { "Models are only available after SAT checks" }
        return KCvc5Model(ctx, cvc5Ctx, cvc5Assertions.flatten(), exprConverter, exprInternalizer)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }
        val cvc5TrackedVars = cvc5TrackedAssertions.flatten().toSet()
        val cvc5FullCore = solver.unsatCore
        val cvc5OnlyTrackedAssertions = cvc5FullCore.filter { it in cvc5TrackedVars }
        val cvc5UnsatAssumptions = solver.unsatAssumptions

        val cvc5UnsatCore = cvc5OnlyTrackedAssertions + cvc5UnsatAssumptions

        with(exprConverter) { cvc5UnsatCore.map { it.convertExpr() } }
    }

    override fun close() {
        cvc5Ctx.close()
        solver.close()
    }

    private fun Result.processCheckResult() = when {
        isSat -> KSolverStatus.SAT
        isUnsat -> KSolverStatus.UNSAT
        isUnknown || isNull -> KSolverStatus.UNKNOWN
        else -> KSolverStatus.UNKNOWN
    }.also {
        lastCheckStatus = it
        if (it == KSolverStatus.UNKNOWN) cvc5LastUnknownExplanation = this.unknownExplanation
    }

    private fun Solver.updateTimeout(timeout: Duration) {
        val cvc5Timeout = if (timeout == Duration.INFINITE) 0 else timeout.toInt(DurationUnit.MILLISECONDS)
        setOption("tlimit-per", cvc5Timeout.toString())
    }

    private inline fun <reified T> cvc5Try(body: () -> T): T = try {
        body()
    } catch (ex: CVC5ApiException) {
        throw KSolverException(ex)
    }

    companion object {
        init {
            System.setProperty("cvc5.skipLibraryLoad", "true")
            NativeLibraryLoader.load { os ->
                when (os) {
                    NativeLibraryLoader.OS.LINUX -> listOf("libcvc5", "libcvc5jni")
                    NativeLibraryLoader.OS.MACOS -> TODO("add macOS cvc5 libs")
                    NativeLibraryLoader.OS.WINDOWS -> TODO("add Windows cvc5 libs")
                }
            }
        }
    }
}
