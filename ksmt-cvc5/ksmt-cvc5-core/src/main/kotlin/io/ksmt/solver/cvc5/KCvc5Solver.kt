package io.ksmt.solver.cvc5

import io.github.cvc5.CVC5ApiException
import io.github.cvc5.Result
import io.github.cvc5.Solver
import io.github.cvc5.TermManager
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.model.KNativeSolverModel
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.library.NativeLibraryLoaderUtils
import kotlin.time.Duration
import kotlin.time.DurationUnit

open class KCvc5Solver(private val ctx: KContext) : KSolver<KCvc5SolverConfiguration> {
    private val termManager = KCvc5TermManager()
    private val solver = termManager.builder { Solver(this) }
    private val cvc5Ctx = KCvc5Context(termManager, ctx)

    private val exprInternalizer by lazy { createExprInternalizer(cvc5Ctx) }

    private val currentScope: UInt
        get() = cvc5TrackedAssertions.lastIndex.toUInt()

    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var lastReasonOfUnknown: String? = null
    private var lastCvcModel: KCvc5Model? = null
    private var lastModel: KModel? = null

    private var cvc5LastAssumptions: KCvc5TermMap<KExpr<KBoolSort>>? = null

    private var cvc5CurrentLevelTrackedAssertions = KCvc5TermMap<KExpr<KBoolSort>>()
    private val cvc5TrackedAssertions = mutableListOf(cvc5CurrentLevelTrackedAssertions)

    init {
        solver.setOption("produce-models", "true")
        solver.setOption("produce-unsat-cores", "true")
        /**
         * Allow floating-point sorts of all sizes, rather than
         * only Float32 (8/24) or Float64 (11/53) (experimental in cvc5 1.0.2)
         */
        solver.setOption("fp-exp", "true")
    }

    open fun createExprInternalizer(cvc5Ctx: KCvc5Context): KCvc5ExprInternalizer =
        KCvc5ExprInternalizer(cvc5Ctx, solver)

    override fun configure(configurator: KCvc5SolverConfiguration.() -> Unit) {
        KCvc5SolverConfigurationImpl(solver).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = cvc5Try {
        ctx.ensureContextMatch(expr)

        val cvc5Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.assertFormula(cvc5Expr)

        cvc5Ctx.assertPendingAxioms(solver)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) {
        ctx.ensureContextMatch(expr)

        val trackVarApp = ctx.mkFreshConst("track", ctx.boolSort)
        val cvc5TrackVar = with(exprInternalizer) { trackVarApp.internalizeExpr() }
        val trackedExpr = with(ctx) { trackVarApp implies expr }

        cvc5CurrentLevelTrackedAssertions[cvc5TrackVar] = expr

        assert(trackedExpr)
        solver.assertFormula(cvc5TrackVar)
    }

    override fun push() = solver.push().also {
        cvc5CurrentLevelTrackedAssertions = KCvc5TermMap()
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

    override fun check(timeout: Duration): KSolverStatus = cvc5TryCheck {
        solver.updateTimeout(timeout)
        solver.checkSat().processCheckResult()
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = cvc5TryCheck {
        ctx.ensureContextMatch(assumptions)

        val lastAssumptions = KCvc5TermMap<KExpr<KBoolSort>>().also { cvc5LastAssumptions = it }
        val cvc5Assumptions = Array(assumptions.size) { idx ->
            val assumedExpr = assumptions[idx]
            with(exprInternalizer) {
                assumedExpr.internalizeExpr().also {
                    lastAssumptions[it] = assumedExpr
                }
            }
        }

        solver.updateTimeout(timeout)
        solver.checkSatAssuming(cvc5Assumptions).processCheckResult()
    }

    override fun reasonOfUnknown(): String = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) {
            "Unknown reason is only available after UNKNOWN checks"
        }
        lastReasonOfUnknown ?: "no explanation"
    }

    override fun model(): KModel = cvc5Try {
        require(lastCheckStatus == KSolverStatus.SAT) { "Models are only available after SAT checks" }
        lastModel?.let { return it }

        val cvcModel = KCvc5Model(
            ctx,
            cvc5Ctx,
            solver,
            exprInternalizer,
            cvc5Ctx.declarations().flatMapTo(hashSetOf()) { it },
            cvc5Ctx.uninterpretedSorts().flatMapTo(hashSetOf()) { it },
        )

        return KNativeSolverModel(cvcModel).also {
            lastCvcModel = cvcModel
            lastModel = it
        }
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }

        val cvc5FullCore = solver.unsatCore.onEach { termManager.registerPointer(it) }

        val trackedTerms = KCvc5TermMap<KExpr<KBoolSort>>()
        cvc5TrackedAssertions.forEach { frame ->
            trackedTerms.putAll(frame)
        }
        cvc5LastAssumptions?.also { trackedTerms.putAll(it) }

        cvc5FullCore.mapNotNull { trackedTerms[it] }
    }

    override fun close() {
        cvc5CurrentLevelTrackedAssertions.clear()
        cvc5TrackedAssertions.clear()

        cvc5Ctx.close()
        termManager.close()
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
        if (it == KSolverStatus.UNKNOWN) {
            lastReasonOfUnknown = this.unknownExplanation?.name
        }
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

    private inline fun cvc5TryCheck(body: () -> KSolverStatus): KSolverStatus = try {
        invalidateSolverState()
        body()
    } catch (ex: CVC5ApiException) {
        lastReasonOfUnknown = ex.message
        KSolverStatus.UNKNOWN.also { lastCheckStatus = it }
    }

    private fun invalidateSolverState() {
        /**
         * Cvc5 model is only valid until the next check-sat call.
         * */
        lastCvcModel?.markInvalid()
        lastCvcModel = null
        lastModel = null

        lastCheckStatus = KSolverStatus.UNKNOWN
        lastReasonOfUnknown = null

        cvc5LastAssumptions = null
    }

    companion object {
        init {
            if (System.getProperty("cvc5.skipLibraryLoad") != "true") {
                NativeLibraryLoaderUtils.load<KCvc5NativeLibraryLoader>()
                System.setProperty("cvc5.skipLibraryLoad", "true")
            }
        }
    }
}
