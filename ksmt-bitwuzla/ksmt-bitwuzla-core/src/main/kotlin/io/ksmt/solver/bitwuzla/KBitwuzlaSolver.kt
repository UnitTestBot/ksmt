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
import kotlin.time.DurationUnit

open class KBitwuzlaSolver(private val ctx: KContext) : KSolver<KBitwuzlaSolverConfiguration> {
    protected val bitwuzlaOptions = Native.bitwuzlaOptionsNew()

    val bitwuzla by lazy { Native.bitwuzlaNew(bitwuzlaOptions).also { bitwuzlaInitialized = true } }
    private var bitwuzlaInitialized = false

    init {
        Native.bitwuzlaSetOption(bitwuzlaOptions, BitwuzlaOption.BITWUZLA_OPTION_PRODUCE_MODELS, value = 1)
        Native.bitwuzlaSetOption(bitwuzlaOptions, BitwuzlaOption.BITWUZLA_OPTION_PRODUCE_UNSAT_ASSUMPTIONS, value = 1)
        Native.bitwuzlaSetOption(bitwuzlaOptions, BitwuzlaOption.BITWUZLA_OPTION_PRODUCE_UNSAT_CORES, value = 1)
    }

    open val bitwuzlaCtx by lazy { KBitwuzlaContext(ctx, bitwuzla) }
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

    private var trackedAssertions = mutableListOf<Pair<KExpr<KBoolSort>, BitwuzlaTerm>>()
    private val trackVarsAssertionFrames = arrayListOf(trackedAssertions)

    private fun internalizeAndAssertWithAxioms(expr: KExpr<KBoolSort>) {
        val assertionWithAxioms = with(exprInternalizer) { expr.internalizeAssertion() }

        assertionWithAxioms.axioms.forEach {
            Native.bitwuzlaAssert(bitwuzla, it)
        }
        Native.bitwuzlaAssert(bitwuzla, assertionWithAxioms.assertion)
    }

    override fun configure(configurator: KBitwuzlaSolverConfiguration.() -> Unit) {
        ensureBitwuzlaNotInitialized()
        KBitwuzlaSolverConfigurationImpl(bitwuzlaOptions).configurator()
    }

    override fun assert(expr: KExpr<KBoolSort>) = bitwuzlaCtx.bitwuzlaTry {
        ctx.ensureContextMatch(expr)
        internalizeAndAssertWithAxioms(expr)
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
        Native.bitwuzlaPush(bitwuzla, nlevels = 1)

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

        Native.bitwuzlaPop(bitwuzla, n.toLong())
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

            val internalizedAssumptions: BitwuzlaTermArray
            with(exprInternalizer) {
                internalizedAssumptions = BitwuzlaTermArray(assumptions.size) { assumptions[it].internalize() }

                assumptions.forEachIndexed { idx, assumption ->
                    currentAssumptions.assumeAssumption(assumption, internalizedAssumptions[idx])
                }
            }

            checkWithTimeout(timeout, internalizedAssumptions).processCheckResult()
        }

    private fun checkWithTimeout(timeout: Duration, assumptions: BitwuzlaTermArray): BitwuzlaResult {
        updateTimeout(timeout)
        return Native.bitwuzlaCheckSatAssuming(bitwuzla, assumptions)
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
        val unsatAssumptions = Native.bitwuzlaGetUnsatAssumptions(bitwuzla)
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
        // TODO: force terminate
//        Native.bitwuzlaForceTerminate(bitwuzla)
    }

    override fun close() = bitwuzlaCtx.bitwuzlaTry {
        bitwuzlaCtx.close()
        Native.bitwuzlaDelete(bitwuzla)
        Native.bitwuzlaOptionsDelete(bitwuzlaOptions)
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

    protected fun updateTimeout(timeout: Duration) {
        val bitwuzlaTimeout = if (timeout == Duration.INFINITE) 0 else timeout.toLong(DurationUnit.MILLISECONDS)
        Native.bitwuzlaSetOption(bitwuzlaOptions, BitwuzlaOption.BITWUZLA_OPTION_TIME_LIMIT_PER, bitwuzlaTimeout)
    }

    private fun ensureBitwuzlaNotInitialized() = check(!bitwuzlaInitialized) { "Bitwuzla has already initialized." }

    private inner class TrackedAssumptions {
        private val assumedExprs = arrayListOf<Pair<KExpr<KBoolSort>, BitwuzlaTerm>>()

        fun assumeTrackedAssertion(trackedAssertion: Pair<KExpr<KBoolSort>, BitwuzlaTerm>) {
            assumedExprs += trackedAssertion
        }

        fun assumeAssumption(expr: KExpr<KBoolSort>, term: BitwuzlaTerm) =
            assumeTrackedAssertion(expr to term)

        fun resolveUnsatCore(unsatAssumptions: BitwuzlaTermArray): List<KExpr<KBoolSort>> {
            val unsatCoreTerms = LongOpenHashSet(unsatAssumptions)
            return assumedExprs.mapNotNull { (expr, term) -> expr.takeIf { unsatCoreTerms.contains(term) } }
        }
    }
}
