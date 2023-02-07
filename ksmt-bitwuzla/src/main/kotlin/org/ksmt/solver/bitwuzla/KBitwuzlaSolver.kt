package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.mkFreshConst
import kotlin.time.Duration

open class KBitwuzlaSolver(private val ctx: KContext) : KSolver<KBitwuzlaSolverConfiguration> {
    open val bitwuzlaCtx = KBitwuzlaContext()
    open val exprInternalizer: KBitwuzlaExprInternalizer by lazy {
        KBitwuzlaExprInternalizer(bitwuzlaCtx)
    }
    open val exprConverter: KBitwuzlaExprConverter by lazy {
        KBitwuzlaExprConverter(ctx, bitwuzlaCtx)
    }
    private var lastCheckStatus = KSolverStatus.UNKNOWN

    init {
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_MODELS, value = 1)
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_UNSAT_CORES, value = 1)
    }

    private var currentLevelTrackedAssertions = hashSetOf<BitwuzlaTerm>()
    private val trackedAssertions = mutableListOf(currentLevelTrackedAssertions)

    override fun configure(configurator: KBitwuzlaSolverConfiguration.() -> Unit) {
        KBitwuzlaSolverConfigurationImpl(bitwuzlaCtx.bitwuzla).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = bitwuzlaCtx.bitwuzlaTry {
        ctx.ensureContextMatch(expr)

        val assertionWithAxioms = with(exprInternalizer) { expr.internalizeAssertion() }
        assertionWithAxioms.axioms.forEach {
            Native.bitwuzlaAssert(bitwuzlaCtx.bitwuzla, it)
        }
        Native.bitwuzlaAssert(bitwuzlaCtx.bitwuzla, assertionWithAxioms.assertion)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort> = bitwuzlaCtx.bitwuzlaTry {
        ctx.ensureContextMatch(expr)

        val trackVar = with(ctx) { boolSort.mkFreshConst("track") }
        val trackedExpr = with(ctx) { !trackVar or expr }
        val assertionWithAxioms = with(exprInternalizer) { trackedExpr.internalizeAssertion() }
        val trackVarTerm = with(exprInternalizer) { trackVar.internalize() }

        currentLevelTrackedAssertions += trackVarTerm

        assertionWithAxioms.axioms.forEach {
            Native.bitwuzlaAssert(bitwuzlaCtx.bitwuzla, it)
        }
        Native.bitwuzlaAssert(bitwuzlaCtx.bitwuzla, assertionWithAxioms.assertion)
        Native.bitwuzlaAssert(bitwuzlaCtx.bitwuzla, trackVarTerm)

        trackVar
    }

    override fun push(): Unit = bitwuzlaCtx.bitwuzlaTry {
        Native.bitwuzlaPush(bitwuzlaCtx.bitwuzla, nlevels = 1)

        currentLevelTrackedAssertions = hashSetOf()
        trackedAssertions.add(currentLevelTrackedAssertions)
    }

    override fun pop(n: UInt): Unit = bitwuzlaCtx.bitwuzlaTry {
        val currentLevel = trackedAssertions.lastIndex.toUInt()
        require(n <= currentLevel) {
            "Cannot pop $n scope levels because current scope level is $currentLevel"
        }

        if (n == 0u) return

        repeat(n.toInt()) { trackedAssertions.removeLast() }
        currentLevelTrackedAssertions = trackedAssertions.last()

        Native.bitwuzlaPop(bitwuzlaCtx.bitwuzla, n.toInt())
    }

    override fun check(timeout: Duration): KSolverStatus = bitwuzlaCtx.bitwuzlaTry {
        checkWithTimeout(timeout).processCheckResult()
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        bitwuzlaCtx.bitwuzlaTry {
            ctx.ensureContextMatch(assumptions)

            val bitwuzlaAssumptions = with(exprInternalizer) {
                assumptions.map { it.internalize() }
            }
            bitwuzlaAssumptions.forEach {
                Native.bitwuzlaAssume(bitwuzlaCtx.bitwuzla, it)
            }

            checkWithTimeout(timeout).processCheckResult()
        }

    private fun checkWithTimeout(timeout: Duration): BitwuzlaResult = if (timeout.isInfinite()) {
        Native.bitwuzlaCheckSatResult(bitwuzlaCtx.bitwuzla)
    } else {
        Native.bitwuzlaCheckSatTimeoutResult(bitwuzlaCtx.bitwuzla, timeout.inWholeMilliseconds)
    }

    override fun model(): KModel = bitwuzlaCtx.bitwuzlaTry {
        require(lastCheckStatus == KSolverStatus.SAT) { "Model are only available after SAT checks" }
        return KBitwuzlaModel(ctx, bitwuzlaCtx, exprInternalizer, exprConverter)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = bitwuzlaCtx.bitwuzlaTry {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }

        val fullCore = Native.bitwuzlaGetUnsatCore(bitwuzlaCtx.bitwuzla)
        val trackVars = trackedAssertions.flatten().toSet()
        val onlyTrackedAssertions = fullCore.filter { it in trackVars }
        val unsatAssumptions = Native.bitwuzlaGetUnsatAssumptions(bitwuzlaCtx.bitwuzla)
        val unsatCore = onlyTrackedAssertions + unsatAssumptions.toList()

        return with(exprConverter) { unsatCore.map { it.convertExpr(ctx.boolSort) } }
    }

    override fun reasonOfUnknown(): String = bitwuzlaCtx.bitwuzlaTry {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) {
            "Unknown reason is only available after UNKNOWN checks"
        }

        // There is no way to retrieve reason of unknown from Bitwuzla in general case.
        return "unknown"
    }

    override fun close() = bitwuzlaCtx.bitwuzlaTry {
        bitwuzlaCtx.close()
    }

    private fun BitwuzlaResult.processCheckResult() = when (this) {
        BitwuzlaResult.BITWUZLA_SAT -> KSolverStatus.SAT
        BitwuzlaResult.BITWUZLA_UNSAT -> KSolverStatus.UNSAT
        BitwuzlaResult.BITWUZLA_UNKNOWN -> KSolverStatus.UNKNOWN
    }.also { lastCheckStatus = it }
}
