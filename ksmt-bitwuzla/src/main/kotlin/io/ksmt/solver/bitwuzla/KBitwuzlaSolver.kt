package io.ksmt.solver.bitwuzla

import it.unimi.dsi.fastutil.longs.LongOpenHashSet
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaNativeException
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTerm
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaTermArray
import org.ksmt.solver.bitwuzla.bindings.Native
import io.ksmt.sort.KBoolSort
import kotlin.time.Duration

open class KBitwuzlaSolver(private val ctx: KContext) : KSolver<KBitwuzlaSolverConfiguration> {
    open val bitwuzlaCtx = KBitwuzlaContext(ctx)
    open val exprInternalizer: KBitwuzlaExprInternalizer by lazy {
        KBitwuzlaExprInternalizer(bitwuzlaCtx)
    }
    open val exprConverter: KBitwuzlaExprConverter by lazy {
        KBitwuzlaExprConverter(ctx, bitwuzlaCtx)
    }

    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var lastReasonOfUnknown: String? = null
    private var lastAssumptions: TrackedAssumptions? = null
    private var lastModel: KBitwuzlaModel? = null

    init {
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_INCREMENTAL, value = 1)
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_MODELS, value = 1)
    }

    private var trackedAssertions = mutableListOf<Pair<KExpr<KBoolSort>, BitwuzlaTerm>>()
    private val trackVarsAssertionFrames = arrayListOf(trackedAssertions)

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

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = bitwuzlaCtx.bitwuzlaTry {
        ctx.ensureContextMatch(expr)

        val trackVarExpr = ctx.mkFreshConst("track", ctx.boolSort)
        val trackedExpr = with(ctx) { !trackVarExpr or expr }

        assert(trackedExpr)

        val trackVarTerm = with(exprInternalizer) { trackVarExpr.internalize() }
        trackedAssertions += expr to trackVarTerm
    }

    override fun push(): Unit = bitwuzlaCtx.bitwuzlaTry {
        Native.bitwuzlaPush(bitwuzlaCtx.bitwuzla, nlevels = 1)

        trackedAssertions = trackedAssertions.toMutableList()
        trackVarsAssertionFrames.add(trackedAssertions)

        bitwuzlaCtx.createNestedDeclarationScope()
    }

    override fun pop(n: UInt): Unit = bitwuzlaCtx.bitwuzlaTry {
        val currentLevel = trackVarsAssertionFrames.lastIndex.toUInt()
        require(n <= currentLevel) {
            "Cannot pop $n scope levels because current scope level is $currentLevel"
        }

        if (n == 0u) return

        repeat(n.toInt()) {
            trackVarsAssertionFrames.removeLast()
            bitwuzlaCtx.popDeclarationScope()
        }

        trackedAssertions = trackVarsAssertionFrames.last()

        Native.bitwuzlaPop(bitwuzlaCtx.bitwuzla, n.toInt())
    }

    override fun check(timeout: Duration): KSolverStatus =
        checkWithAssumptions(emptyList(), timeout)

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        bitwuzlaTryCheck {
            ctx.ensureContextMatch(assumptions)

            val currentAssumptions = TrackedAssumptions().also { lastAssumptions = it }

            trackedAssertions.forEach {
                currentAssumptions.assumeTrackedAssertion(it)
            }

            with(exprInternalizer) {
                assumptions.forEach {
                    currentAssumptions.assumeAssumption(it, it.internalize())
                }
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
        val model = lastModel ?: KBitwuzlaModel(
            ctx, bitwuzlaCtx, exprConverter,
            bitwuzlaCtx.declarations(),
            bitwuzlaCtx.uninterpretedSortsWithRelevantDecls()
        )
        lastModel = model
        model
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = bitwuzlaCtx.bitwuzlaTry {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }
        val unsatAssumptions = Native.bitwuzlaGetUnsatAssumptions(bitwuzlaCtx.bitwuzla)
        lastAssumptions?.resolveUnsatCore(unsatAssumptions) ?: emptyList()
    }

    override fun reasonOfUnknown(): String = bitwuzlaCtx.bitwuzlaTry {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) {
            "Unknown reason is only available after UNKNOWN checks"
        }

        // There is no way to retrieve reason of unknown from Bitwuzla in general case.
        return lastReasonOfUnknown ?: "unknown"
    }

    override fun interrupt() = bitwuzlaCtx.bitwuzlaTry {
        Native.bitwuzlaForceTerminate(bitwuzlaCtx.bitwuzla)
    }

    override fun close() = bitwuzlaCtx.bitwuzlaTry {
        bitwuzlaCtx.close()
    }

    private fun BitwuzlaResult.processCheckResult() = when (this) {
        BitwuzlaResult.BITWUZLA_SAT -> KSolverStatus.SAT
        BitwuzlaResult.BITWUZLA_UNSAT -> KSolverStatus.UNSAT
        BitwuzlaResult.BITWUZLA_UNKNOWN -> KSolverStatus.UNKNOWN
    }.also { lastCheckStatus = it }

    private fun invalidateSolverState() {
        /**
         * Bitwuzla model is only valid until the next check-sat call.
         * */
        lastModel?.markInvalid()
        lastModel = null

        lastCheckStatus = KSolverStatus.UNKNOWN
        lastReasonOfUnknown = null

        lastAssumptions = null
    }

    private inline fun bitwuzlaTryCheck(body: () -> KSolverStatus): KSolverStatus = try {
        invalidateSolverState()
        body()
    } catch (ex: BitwuzlaNativeException) {
        lastReasonOfUnknown = ex.message
        KSolverStatus.UNKNOWN.also { lastCheckStatus = it }
    }

    private inner class TrackedAssumptions {
        private val assumedExprs = arrayListOf<Pair<KExpr<KBoolSort>, BitwuzlaTerm>>()

        fun assumeTrackedAssertion(trackedAssertion: Pair<KExpr<KBoolSort>, BitwuzlaTerm>) {
            assumedExprs.add(trackedAssertion)
            Native.bitwuzlaAssume(bitwuzlaCtx.bitwuzla, trackedAssertion.second)
        }

        fun assumeAssumption(expr: KExpr<KBoolSort>, term: BitwuzlaTerm) =
            assumeTrackedAssertion(expr to term)

        fun resolveUnsatCore(unsatAssumptions: BitwuzlaTermArray): List<KExpr<KBoolSort>> {
            val unsatCoreTerms = LongOpenHashSet(unsatAssumptions)
            return assumedExprs.mapNotNull { (expr, term) -> expr.takeIf { unsatCoreTerms.contains(term) } }
        }
    }
}
