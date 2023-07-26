package io.ksmt.solver.cvc5

import io.github.cvc5.CVC5ApiException
import io.github.cvc5.Result
import io.github.cvc5.Solver
import io.github.cvc5.Term
import io.ksmt.KContext
import io.ksmt.expr.KApp
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.NativeLibraryLoader
import java.util.TreeMap
import kotlin.time.Duration
import kotlin.time.DurationUnit

abstract class KCvc5SolverBase internal constructor(
    protected val ctx: KContext
) : KSolver<KCvc5SolverConfiguration> {

    protected abstract val currentScope: UInt

    protected val solver = Solver().apply { configureInitially() }
    protected abstract val cvc5Ctx: KCvc5Context
    protected val exprInternalizer by lazy { createExprInternalizer(cvc5Ctx) }

    protected var lastCheckStatus = KSolverStatus.UNKNOWN
    protected var lastReasonOfUnknown: String? = null
    protected var lastModel: KCvc5Model? = null

    // use TreeMap for cvc5 Term (hashcode not implemented)
    protected var lastCvc5Assumptions: TreeMap<Term, KExpr<KBoolSort>>? = null

    open fun createExprInternalizer(cvc5Ctx: KCvc5Context): KCvc5ExprInternalizer = KCvc5ExprInternalizer(cvc5Ctx)

    private fun Solver.configureInitially() {
        setOption("produce-models", "true")
        setOption("produce-unsat-cores", "true")
        /**
         * Allow floating-point sorts of all sizes, rather than
         * only Float32 (8/24) or Float64 (11/53)
         */
        setOption("fp-exp", "true")
    }

    override fun configure(configurator: KCvc5SolverConfiguration.() -> Unit) {
        KCvc5SolverConfigurationImpl(solver).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = cvc5Try {
        ctx.ensureContextMatch(expr)

        val cvc5Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.assertFormula(cvc5Expr)
        cvc5Ctx.assertPendingAxioms(solver)
    }

    protected abstract fun saveTrackedAssertion(track: Term, trackedExpr: KExpr<KBoolSort>)
    protected abstract fun findTrackedExprByTrack(track: Term): KExpr<KBoolSort>?

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = cvc5Try {
        ctx.ensureContextMatch(expr)

        val trackVarApp = createTrackVarApp()
        val cvc5TrackVar = with(exprInternalizer) { trackVarApp.internalizeExpr() }
        val trackedExpr = with(ctx) { trackVarApp implies expr }
        assert(trackedExpr)
        solver.assertFormula(cvc5TrackVar)
        saveTrackedAssertion(cvc5TrackVar, expr)
    }

    override fun push() = cvc5Try {
        solver.push()
        cvc5Ctx.push()
    }

    override fun pop(n: UInt) = cvc5Try {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }

        if (n == 0u) return
        solver.pop(n.toInt())
        cvc5Ctx.pop(n)
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

        val lastAssumptions = TreeMap<Term, KExpr<KBoolSort>>().also { lastCvc5Assumptions = it }
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

    protected open fun freshModel(): KCvc5Model = KCvc5Model(
        ctx,
        cvc5Ctx,
        exprInternalizer,
        cvc5Ctx.declarations(),
        cvc5Ctx.uninterpretedSorts(),
    )

    override fun model(): KModel = cvc5Try {
        require(lastCheckStatus == KSolverStatus.SAT) { "Models are only available after SAT checks" }
        val model = lastModel ?: freshModel()
        model.also { lastModel = it }
    }

    override fun reasonOfUnknown(): String = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) {
            "Unknown reason is only available after UNKNOWN checks"
        }
        lastReasonOfUnknown ?: "no explanation"
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        val cvc5FullCore = cvc5UnsatCore()

        val unsatCore = mutableListOf<KExpr<KBoolSort>>()

        cvc5FullCore.forEach { unsatCoreTerm ->
            lastCvc5Assumptions?.get(unsatCoreTerm)?.also { unsatCore += it }
                ?: findTrackedExprByTrack(unsatCoreTerm)?.also { unsatCore += it }
        }
        return unsatCore
    }

    protected fun cvc5UnsatCore(): Array<Term> = cvc5Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }
        solver.unsatCore
    }

    override fun close() {
        cvc5Ctx.close()
        solver.close()
    }

    /*
    there is no method to interrupt cvc5
    */
    override fun interrupt() = Unit

    protected fun createTrackVarApp(): KApp<KBoolSort, *> = ctx.mkFreshConst("track", ctx.boolSort)

    protected fun Result.processCheckResult() = when {
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

    protected fun Solver.updateTimeout(timeout: Duration) {
        val cvc5Timeout = if (timeout == Duration.INFINITE) 0 else timeout.toInt(DurationUnit.MILLISECONDS)
        setOption("tlimit-per", cvc5Timeout.toString())
    }

    protected inline fun <reified T> cvc5Try(body: () -> T): T = try {
        body()
    } catch (ex: CVC5ApiException) {
        throw KSolverException(ex)
    }

    protected inline fun cvc5TryCheck(body: () -> KSolverStatus): KSolverStatus = try {
        invalidateSolverState()
        body()
    } catch (ex: CVC5ApiException) {
        lastReasonOfUnknown = ex.message
        KSolverStatus.UNKNOWN.also { lastCheckStatus = it }
    }

    protected fun invalidateSolverState() {
        /**
         * Cvc5 model is only valid until the next check-sat call.
         * */
        lastModel?.markInvalid()
        lastModel = null
        lastCheckStatus = KSolverStatus.UNKNOWN
        lastReasonOfUnknown = null
        lastCvc5Assumptions = null
    }

    companion object {
        internal fun ensureCvc5LibLoaded() {
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

        init {
            ensureCvc5LibLoaded()
        }
    }
}
