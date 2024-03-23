package io.ksmt.solver.yices

import com.sri.yices.Config
import com.sri.yices.Context
import com.sri.yices.Status
import com.sri.yices.YicesException
import it.unimi.dsi.fastutil.ints.IntArrayList
import it.unimi.dsi.fastutil.ints.IntOpenHashSet
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.model.KNativeSolverModel
import io.ksmt.sort.KBoolSort
import java.util.Timer
import java.util.TimerTask
import kotlin.time.Duration

class KYicesSolver(private val ctx: KContext) : KSolver<KYicesSolverConfiguration> {
    private val yicesCtx = KYicesContext()

    private val config = Config()
    private val nativeContext by lazy {
        Context(config).also {
            config.close()
        }
    }

    private val exprInternalizer: KYicesExprInternalizer by lazy {
        KYicesExprInternalizer(yicesCtx)
    }
    private val exprConverter: KYicesExprConverter by lazy {
        KYicesExprConverter(ctx, yicesCtx)
    }

    private var lastAssumptions: TrackedAssumptions? = null
    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var lastModel: KModel? = null
    private var lastReasonOfUnknown: String? = null

    private var currentLevelTrackedAssertions = mutableListOf<Pair<KExpr<KBoolSort>, YicesTerm>>()
    private val trackedAssertions = mutableListOf(currentLevelTrackedAssertions)

    private val timer = Timer()

    override fun configure(configurator: KYicesSolverConfiguration.() -> Unit) {
        require(config.isActive) {
            "Solver instance has already been created"
        }

        KYicesSolverConfigurationImpl(config).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = yicesTry {
        ctx.ensureContextMatch(expr)

        val yicesExpr = with(exprInternalizer) { expr.internalize() }
        nativeContext.assertFormula(yicesExpr)
    }

    override fun assert(exprs: List<KExpr<KBoolSort>>) = yicesTry {
        ctx.ensureContextMatch(exprs)

        val yicesExprs = with(exprInternalizer) {
            IntArray(exprs.size) { idx -> exprs[idx].internalize() }
        }
        nativeContext.assertFormulas(yicesExprs)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = yicesTry {
        ctx.ensureContextMatch(expr)

        val trackVarExpr = ctx.mkFreshConst("track", ctx.boolSort)
        val trackedExpr = with(ctx) { !trackVarExpr or expr }

        assert(trackedExpr)

        val yicesTrackVar = with(exprInternalizer) { trackVarExpr.internalize() }
        currentLevelTrackedAssertions += expr to yicesTrackVar
    }

    override fun push(): Unit = yicesTry {
        nativeContext.push()

        currentLevelTrackedAssertions = mutableListOf()
        trackedAssertions.add(currentLevelTrackedAssertions)
    }

    override fun pop(n: UInt) = yicesTry {
        val currentScope = trackedAssertions.lastIndex.toUInt()
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }

        if (n == 0u) return

        repeat(n.toInt()) {
            nativeContext.pop()
            trackedAssertions.removeLast()
        }
        currentLevelTrackedAssertions = trackedAssertions.last()
    }

    override fun check(timeout: Duration): KSolverStatus = yicesTryCheck {
        if (trackedAssertions.any { it.isNotEmpty() }) {
            return checkWithAssumptions(emptyList(), timeout)
        }

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

        trackedAssertions.forEach { frame ->
            frame.forEach { assertion ->
                yicesAssumptions.assumeTrackedAssertion(assertion)
            }
        }

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
        lastModel?.let { return it }

        val yicesModel = KYicesModel(nativeContext.model, ctx, yicesCtx, exprInternalizer, exprConverter)
        return KNativeSolverModel(yicesModel).also {
            lastModel = it
        }
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

    private inline fun <T> yicesTry(body: () -> T): T = try {
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
        lastModel = null
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
}
