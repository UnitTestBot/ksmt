package org.ksmt.solver.cvc5

import io.github.cvc5.CVC5ApiException
import io.github.cvc5.Result
import io.github.cvc5.Solver
import io.github.cvc5.Term
import io.github.cvc5.UnknownExplanation
import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.KSolverUnsupportedFeatureException
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.NativeLibraryLoader
import java.util.TreeSet
import kotlin.time.Duration
import kotlin.time.DurationUnit

open class KCvc5Solver(private val ctx: KContext) : KSolver<KCvc5SolverConfiguration> {
    private val solver = Solver()
    private val cvc5Ctx = KCvc5Context(solver, ctx)

    private val exprInternalizer by lazy { createExprInternalizer(cvc5Ctx) }
    private val exprConverter: KCvc5ExprConverter by lazy { createExprConverter(cvc5Ctx) }

    private val currentScope: UInt
        get() = cvc5TrackedAssertions.lastIndex.toUInt()

    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var cvc5LastUnknownExplanation: UnknownExplanation? = null

    private var cvc5CurrentLevelTrackedAssertions = mutableListOf<Term>()
    private val cvc5TrackedAssertions = mutableListOf(cvc5CurrentLevelTrackedAssertions)

    private var cvc5LastAssumptions = TreeSet<Term>()

    init {
        solver.setOption("produce-models", "true")
        solver.setOption("produce-unsat-cores", "true")
        /**
         * Allow floating-point sorts of all sizes, rather than
         * only Float32 (8/24) or Float64 (11/53) (experimental in cvc5 1.0.2)
         */
        solver.setOption("fp-exp", "true")
    }

    open fun createExprInternalizer(cvc5Ctx: KCvc5Context): KCvc5ExprInternalizer = KCvc5ExprInternalizer(cvc5Ctx)

    open fun createExprConverter(cvc5Ctx: KCvc5Context): KCvc5ExprConverter = KCvc5ExprConverter(ctx, cvc5Ctx)

    override fun configure(configurator: KCvc5SolverConfiguration.() -> Unit) {
        KCvc5SolverConfigurationImpl(solver).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = cvc5Try {
        ctx.ensureContextMatch(expr)

        val cvc5Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.assertFormula(cvc5Expr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) {
        ctx.ensureContextMatch(expr, trackVar)

        val trackVarApp = trackVar.apply()
        val cvc5TrackVar = with(exprInternalizer) { trackVarApp.internalizeExpr() }
        val trackedExpr = with(ctx) { trackVarApp implies expr }

        cvc5CurrentLevelTrackedAssertions.add(cvc5TrackVar)

        assert(trackedExpr)
        solver.assertFormula(cvc5TrackVar)
    }

    override fun push() = solver.push().also {
        cvc5CurrentLevelTrackedAssertions = mutableListOf()
        cvc5TrackedAssertions.add(cvc5CurrentLevelTrackedAssertions)

        cvc5Ctx.push()
    }

    override fun pop(n: UInt) {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }

        if (n == 0u) return

        repeat(n.toInt()) {
            cvc5TrackedAssertions.removeLast()
        }
        cvc5CurrentLevelTrackedAssertions = cvc5TrackedAssertions.last()

        cvc5Ctx.pop(n)
        solver.pop(n.toInt())
    }

    override fun check(timeout: Duration): KSolverStatus = cvc5Try {
        cvc5LastAssumptions = TreeSet()
        solver.updateTimeout(timeout)
        solver.checkSat().processCheckResult()
    }

    override fun reasonOfUnknown(): String = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) { "Unknown reason is only available after UNKNOWN checks" }
        cvc5LastUnknownExplanation?.name ?: "no explanation"
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus = cvc5Try {
        ctx.ensureContextMatch(assumptions)

        val cvc5Assumptions = with(exprInternalizer) { assumptions.map { it.internalizeExpr() } }.toTypedArray()
        cvc5LastAssumptions = cvc5Assumptions.toCollection(TreeSet())
        solver.updateTimeout(timeout)
        solver.checkSatAssuming(cvc5Assumptions).processCheckResult()
    }

    override fun model(): KModel = cvc5Try {
        require(lastCheckStatus == KSolverStatus.SAT) { "Models are only available after SAT checks" }
        return KCvc5Model(
            ctx,
            cvc5Ctx,
            exprConverter,
            exprInternalizer,
            cvc5Ctx.declarations().flatten().toSet(),
            cvc5Ctx.uninterpretedSorts().flatten().toSet()
        )
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }
        // we need TreeSet here (hashcode not implemented in Term)
        val cvc5TrackedVars = cvc5TrackedAssertions.flatten().toSortedSet()
        val cvc5FullCore = solver.unsatCore
        val cvc5UnsatCore = cvc5FullCore.filter { it in cvc5TrackedVars || it in cvc5LastAssumptions }

        with(exprConverter) { cvc5UnsatCore.map { it.convertExpr() } }
    }

    override fun close() {
        cvc5CurrentLevelTrackedAssertions.clear()
        cvc5TrackedAssertions.clear()

        cvc5Ctx.close()
        solver.close()
    }

    /*
     there are no methods to interrupt cvc5,
     but maybe CVC5ApiRecoverableException can be thrown in someway
    */
    override fun interrupt() = Unit

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
            if (System.getProperty("cvc5.skipLibraryLoad") != "true") {
                NativeLibraryLoader.load { os ->
                    when (os) {
                        NativeLibraryLoader.OS.LINUX -> listOf("libcvc5", "libcvc5jni")
                        NativeLibraryLoader.OS.WINDOWS -> listOf("libcvc5", "libcvc5jni")
                        NativeLibraryLoader.OS.MACOS -> listOf("libcvc5", "libcvc5jni")
                    }
                }
                System.setProperty("cvc5.skipLibraryLoad", "true")
            }
        }
    }
}
