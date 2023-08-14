package io.ksmt.solver.yices

import com.sri.yices.Config
import com.sri.yices.Context
import com.sri.yices.Status
import com.sri.yices.Yices
import com.sri.yices.YicesException
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.NativeLibraryLoader
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import java.util.Timer
import java.util.TimerTask
import kotlin.time.Duration

abstract class KYicesSolverBase(protected val ctx: KContext) : KSolver<KYicesSolverConfiguration> {
    protected abstract val yicesCtx: KYicesContext
    protected val nativeContext by lazy { Context(config).also { config.close() } }
    protected val config by lazy { Config() }

    protected val exprInternalizer: KYicesExprInternalizer by lazy { KYicesExprInternalizer(yicesCtx) }
    protected val exprConverter: KYicesExprConverter by lazy { KYicesExprConverter(ctx, yicesCtx) }

    private var lastAssumptions: TrackedAssumptions? = null
    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var lastReasonOfUnknown: String? = null

    protected abstract val currentScope: UInt

    protected abstract fun saveTrackedAssertion(track: YicesTerm, trackedExpr: KExpr<KBoolSort>)
    protected abstract fun collectTrackedAssertions(collector: (Pair<KExpr<KBoolSort>, YicesTerm>) -> Unit)
    protected abstract val hasTrackedAssertions: Boolean

    private val timer = Timer()

    protected fun requireActiveConfig() = require(config.isActive) {
        "Solver instance has already been created"
    }

    override fun assert(expr: KExpr<KBoolSort>) = yicesTry {
        ctx.ensureContextMatch(expr)

        val yicesExpr = with(exprInternalizer) { expr.internalize() }
        nativeContext.assertFormula(yicesExpr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = yicesTry {
        ctx.ensureContextMatch(expr)

        val trackVarExpr = ctx.mkFreshConst("track", ctx.boolSort)
        val trackedExpr = with(ctx) { !trackVarExpr or expr }

        assert(trackedExpr)

        val yicesTrackVar = with(exprInternalizer) { trackVarExpr.internalize() }
        saveTrackedAssertion(yicesTrackVar, expr)
    }

    override fun push(): Unit = yicesTry {
        nativeContext.push()
        yicesCtx.pushAssertionLevel()
    }

    override fun pop(n: UInt) = yicesTry {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }

        if (n == 0u) return

        repeat(n.toInt()) {
            nativeContext.pop()
        }
        yicesCtx.popAssertionLevel(n)
    }

    override fun check(timeout: Duration): KSolverStatus = if (hasTrackedAssertions) {
        checkWithAssumptions(emptyList(), timeout)
    } else yicesTryCheck {
        checkWithTimer(timeout) {
            nativeContext.check()
        }.processCheckResult()
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = yicesTryCheck {
        ctx.ensureContextMatch(assumptions)

        val yicesAssumptions = TrackedAssumptions().also { lastAssumptions = it }

        collectTrackedAssertions(yicesAssumptions::assumeTrackedAssertion)

        with(exprInternalizer) {
            assumptions.forEach { assumedExpr ->
                yicesAssumptions.assumeAssumption(assumedExpr, assumedExpr.internalize())
            }
        }

        checkWithTimer(timeout) {
            nativeContext.checkWithAssumptions(yicesAssumptions.assumedTerms())
        }.processCheckResult()
    }

    override fun model(): KModel = yicesTry {
        require(lastCheckStatus == KSolverStatus.SAT) {
            "Model are only available after SAT checks, current solver status: $lastCheckStatus"
        }
        val model = nativeContext.model

        return KYicesModel(model, ctx, yicesCtx, exprInternalizer, exprConverter)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = yicesTry {
        require(lastCheckStatus == KSolverStatus.UNSAT) {
            "Unsat cores are only available after UNSAT checks"
        }

        lastAssumptions?.resolveUnsatCore(nativeContext.unsatCore) ?: emptyList()
    }

    override fun reasonOfUnknown(): String {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) {
            "Unknown reason is only available after UNKNOWN checks"
        }

        // There is no way to retrieve reason of unknown from Yices in general case.
        return lastReasonOfUnknown ?: "unknown"
    }

    override fun interrupt() = yicesTry {
        nativeContext.stopSearch()
    }

    private inline fun <T> checkWithTimer(timeout: Duration, body: () -> T): T {
        val task = StopSearchTask()

        if (timeout.isFinite()) {
            timer.schedule(task, timeout.inWholeMilliseconds)
        }

        return try {
            body()
        } finally {
            task.cancel()
        }
    }

    protected inline fun <T> yicesTry(body: () -> T): T = try {
        body()
    } catch (ex: YicesException) {
        throw KSolverException(ex)
    }

    private inline fun yicesTryCheck(body: () -> KSolverStatus): KSolverStatus = try {
        invalidateSolverState()
        body()
    } catch (ex: YicesException) {
        lastReasonOfUnknown = ex.message
        KSolverStatus.UNKNOWN.also { lastCheckStatus = it }
    }

    private fun invalidateSolverState() {
        lastCheckStatus = KSolverStatus.UNKNOWN
        lastReasonOfUnknown = null
        lastAssumptions = null
    }

    private fun Status.processCheckResult() = when (this) {
        Status.SAT -> KSolverStatus.SAT
        Status.UNSAT -> KSolverStatus.UNSAT
        else -> KSolverStatus.UNKNOWN
    }.also { lastCheckStatus = it }

    override fun close() {
        nativeContext.close()
        yicesCtx.close()
        timer.cancel()
    }

    private inner class StopSearchTask : TimerTask() {
        override fun run() {
            nativeContext.stopSearch()
        }
    }

    private class TrackedAssumptions {
        private val assumedExprs = arrayListOf<Pair<KExpr<KBoolSort>, YicesTerm>>()
        private val assumedTerms = IntArrayList()

        fun assumeTrackedAssertion(trackedAssertion: Pair<KExpr<KBoolSort>, YicesTerm>) {
            assumedExprs.add(trackedAssertion)
            assumedTerms.add(trackedAssertion.second)
        }

        fun assumeAssumption(expr: KExpr<KBoolSort>, term: YicesTerm) =
            assumeTrackedAssertion(expr to term)

        fun assumedTerms(): YicesTermArray {
            assumedTerms.trim() // Elements length now equal to size
            return assumedTerms.elements()
        }

        fun resolveUnsatCore(yicesUnsatCore: YicesTermArray): List<KExpr<KBoolSort>> {
            val unsatCoreTerms = IntOpenHashSet(yicesUnsatCore)
            return assumedExprs.mapNotNull { (expr, term) -> expr.takeIf { unsatCoreTerms.contains(term) } }
        }
    }

    companion object {
        init {
            if (!Yices.isReady()) {
                NativeLibraryLoader.load { os ->
                    when (os) {
                        NativeLibraryLoader.OS.LINUX -> listOf("libyices", "libyices2java")
                        NativeLibraryLoader.OS.WINDOWS -> listOf("libyices", "libyices2java")
                        NativeLibraryLoader.OS.MACOS -> listOf("libyices", "libyices2java")
                    }
                }
                Yices.init()
                Yices.setReadyFlag(true)
            }
        }
    }
}
