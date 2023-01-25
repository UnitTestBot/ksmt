package org.ksmt.solver.yices

import com.sri.yices.Config
import com.sri.yices.Context
import com.sri.yices.Status
import com.sri.yices.YicesException
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.mkFreshConst
import java.util.*
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
        KYicesExprInternalizer(ctx, yicesCtx)
    }
    private val exprConverter: KYicesExprConverter by lazy {
        KYicesExprConverter(ctx, yicesCtx)
    }

    private var lastCheckStatus = KSolverStatus.UNKNOWN

    private var currentLevelTrackedAssertions = mutableListOf<YicesTerm>()
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

    override fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort> = yicesTry {
        ctx.ensureContextMatch(expr)

        val trackVar = with(ctx) { boolSort.mkFreshConst("track") }
        val trackedExpr = with(ctx) { !trackVar or expr }
        val yicesVar = with(exprInternalizer) { trackVar.internalize() }
        val yicesTrackedExpr = with(exprInternalizer) { trackedExpr.internalize() }

        currentLevelTrackedAssertions += yicesVar

        nativeContext.assertFormula(yicesVar)
        nativeContext.assertFormula(yicesTrackedExpr)

        trackVar
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

    override fun check(timeout: Duration): KSolverStatus {
        return withTimer(timeout) {
            nativeContext.check()
        }.processCheckResult()
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus {
        ctx.ensureContextMatch(assumptions)

        val trackedExpressions = trackedAssertions.flatten()
        val yicesAssumptions = with(exprInternalizer) { assumptions.map { it.internalize() } } +
                trackedExpressions

        return withTimer(timeout) {
            nativeContext
                .checkWithAssumptions(yicesAssumptions.toIntArray())
        }.processCheckResult()
    }

    override fun model(): KModel = yicesTry {
        require(lastCheckStatus == KSolverStatus.SAT) {
            "Model are only available after SAT checks, current solver status: $lastCheckStatus"
        }
        val model = nativeContext.model

        return KYicesModel(model, ctx, exprInternalizer, exprConverter)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = yicesTry {
        require(lastCheckStatus == KSolverStatus.UNSAT) {
            "Unsat cores are only available after UNSAT checks"
        }

        val yicesCore = nativeContext.unsatCore

        with(exprConverter) { yicesCore.map { it.convert() } }
    }

    override fun reasonOfUnknown(): String {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) {
            "Unknown reason is only available after UNKNOWN checks"
        }

        // There is no way to retrieve reason of unknown from Yices in general case.
        return "unknown"
    }

    private inline fun <T> withTimer(timeout: Duration, body: () -> T): T {
        val task = StopSearchTask()

        if (timeout.isFinite())
            timer.schedule(task, timeout.inWholeMilliseconds)

        return try {
            body()
        } catch (ex: YicesException) {
            throw KSolverException(ex)
        } finally {
            task.cancel()
        }
    }

    private inline fun <T> yicesTry(body: () -> T): T = try {
        body()
    } catch (ex: YicesException) {
        throw KSolverException(ex)
    }

    private fun Status.processCheckResult() = when (this) {
        Status.SAT -> KSolverStatus.SAT
        Status.UNSAT -> KSolverStatus.UNSAT
        else -> KSolverStatus.UNKNOWN
    }.also { lastCheckStatus = it }

    override fun close() {
        nativeContext.close()
        yicesCtx.close()
    }

    private inner class StopSearchTask : TimerTask() {
        override fun run() {
            nativeContext.stopSearch()
        }
    }
}
