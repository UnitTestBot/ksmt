package io.ksmt.solver.z3

import com.microsoft.z3.Solver
import com.microsoft.z3.Status
import com.microsoft.z3.Z3Exception
import com.microsoft.z3.solverAssert
import com.microsoft.z3.solverAssertAndTrack
import com.microsoft.z3.solverCheckAssumptions
import com.microsoft.z3.solverGetUnsatCore
import io.ksmt.KContext
import io.ksmt.expr.KExpr
import io.ksmt.solver.KModel
import io.ksmt.solver.KSolver
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverStatus
import io.ksmt.solver.model.KNativeSolverModel
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.library.NativeLibraryLoaderUtils
import it.unimi.dsi.fastutil.longs.Long2ObjectOpenHashMap
import java.lang.ref.PhantomReference
import java.lang.ref.ReferenceQueue
import java.util.*
import kotlin.time.Duration
import kotlin.time.DurationUnit

open class KZ3Solver(private val ctx: KContext) : KSolver<KZ3SolverConfiguration> {
    private val z3Ctx = KZ3Context(ctx)
    private val solver = createSolver()

    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var lastReasonOfUnknown: String? = null
    private var lastModel: KModel? = null
    private var lastUnsatCore: List<KExpr<KBoolSort>>? = null

    private var currentScope: UInt = 0u

    @Suppress("LeakingThis")
    private val contextCleanupActionHandler = registerContextForCleanup(this, z3Ctx)

    private val exprInternalizer by lazy {
        createExprInternalizer(z3Ctx)
    }
    private val exprConverter by lazy {
        createExprConverter(z3Ctx)
    }

    open fun createExprInternalizer(z3Ctx: KZ3Context): KZ3ExprInternalizer = KZ3ExprInternalizer(ctx, z3Ctx)

    open fun createExprConverter(z3Ctx: KZ3Context) = KZ3ExprConverter(ctx, z3Ctx)

    private fun createSolver(): Solver = z3Ctx.nativeContext.mkSolver()

    override fun configure(configurator: KZ3SolverConfiguration.() -> Unit) {
        val params = z3Ctx.nativeContext.mkParams()
        KZ3SolverConfigurationImpl(params).configurator()
        solver.setParameters(params)
    }

    override fun push() {
        solver.push()
        z3Ctx.pushAssertionLevel()
        currentScope++
    }

    override fun pop(n: UInt) {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }
        if (n == 0u) return

        solver.pop(n.toInt())
        repeat(n.toInt()) { z3Ctx.popAssertionLevel() }

        currentScope -= n
    }

    override fun assert(expr: KExpr<KBoolSort>) = z3Try {
        ctx.ensureContextMatch(expr)

        val z3Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.solverAssert(z3Expr)

        z3Ctx.assertPendingAxioms(solver)
    }

    private val trackedAssertions = Long2ObjectOpenHashMap<KExpr<KBoolSort>>()

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = z3Try {
        ctx.ensureContextMatch(expr)

        val trackExpr = ctx.mkFreshConst("track", ctx.boolSort)
        val z3Expr = with(exprInternalizer) { expr.internalizeExpr() }
        val z3TrackVar = with(exprInternalizer) { trackExpr.internalizeExpr() }

        trackedAssertions.put(z3TrackVar, expr)

        solver.solverAssertAndTrack(z3Expr, z3TrackVar)
    }

    override fun check(timeout: Duration): KSolverStatus = z3TryCheck {
        solver.updateTimeout(timeout)
        solver.check().processCheckResult()
    }

    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = z3TryCheck {
        ctx.ensureContextMatch(assumptions)

        val z3Assumptions = with(exprInternalizer) {
            LongArray(assumptions.size) {
                val assumption = assumptions[it]

                /**
                 * Assumptions are trivially unsat and no check-sat is required.
                 * */
                if (assumption == ctx.falseExpr) {
                    lastUnsatCore = listOf(ctx.falseExpr)
                    lastCheckStatus = KSolverStatus.UNSAT
                    return KSolverStatus.UNSAT
                }

                assumption.internalizeExpr()
            }
        }

        solver.updateTimeout(timeout)

        solver.solverCheckAssumptions(z3Assumptions).processCheckResult()
    }

    override fun model(): KModel = z3Try {
        require(lastCheckStatus == KSolverStatus.SAT) {
            "Model are only available after SAT checks, current solver status: $lastCheckStatus"
        }
        lastModel?.let { return it }

        val z3Model = KZ3Model(solver.model, ctx, z3Ctx, exprInternalizer)
        return KNativeSolverModel(z3Model).also {
            lastModel = it
        }
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = z3Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }

        val unsatCore = lastUnsatCore ?: with(exprConverter) {
            val solverUnsatCore = solver.solverGetUnsatCore()
            solverUnsatCore.map { trackedAssertions.get(it) ?: it.convertExpr() }
        }
        lastUnsatCore = unsatCore

        unsatCore
    }

    override fun reasonOfUnknown(): String = z3Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) { "Unknown reason is only available after UNKNOWN checks" }
        lastReasonOfUnknown ?: solver.reasonUnknown
    }

    override fun interrupt() = z3Try {
        solver.interrupt()
    }

    override fun close() {
        unregisterContextCleanup(contextCleanupActionHandler)
        z3Ctx.close()
    }

    private fun Status?.processCheckResult() = when (this) {
        Status.SATISFIABLE -> KSolverStatus.SAT
        Status.UNSATISFIABLE -> KSolverStatus.UNSAT
        Status.UNKNOWN -> KSolverStatus.UNKNOWN
        null -> KSolverStatus.UNKNOWN
    }.also { lastCheckStatus = it }

    private fun Solver.updateTimeout(timeout: Duration) {
        val z3Timeout = if (timeout == Duration.INFINITE) {
            UInt.MAX_VALUE.toInt()
        } else {
            timeout.toInt(DurationUnit.MILLISECONDS)
        }
        val params = z3Ctx.nativeContext.mkParams().apply {
            add("timeout", z3Timeout)
        }
        setParameters(params)
    }

    private inline fun <reified T> z3Try(body: () -> T): T = try {
        body()
    } catch (ex: Z3Exception) {
        throw KSolverException(ex)
    }

    private fun invalidateSolverState() {
        lastReasonOfUnknown = null
        lastCheckStatus = KSolverStatus.UNKNOWN
        lastModel = null
        lastUnsatCore = null
    }

    private inline fun z3TryCheck(body: () -> KSolverStatus): KSolverStatus = try {
        invalidateSolverState()
        body()
    } catch (ex: Z3Exception) {
        lastReasonOfUnknown = ex.message
        KSolverStatus.UNKNOWN.also { lastCheckStatus = it }
    }

    companion object {
        init {
            System.setProperty("z3.skipLibraryLoad", "true")
            NativeLibraryLoaderUtils.load<KZ3NativeLibraryLoader>()
        }

        private val cleanupHandlers = ReferenceQueue<KZ3Solver>()
        private val contextForCleanup = IdentityHashMap<PhantomReference<KZ3Solver>, KZ3Context>()

        /** Ensure Z3 native context is closed and all native memory is released.
         * */
        private fun registerContextForCleanup(solver: KZ3Solver, context: KZ3Context): PhantomReference<KZ3Solver> {
            cleanupStaleContexts()
            val cleanupHandler = PhantomReference(solver, cleanupHandlers)
            contextForCleanup[cleanupHandler] = context

            return cleanupHandler
        }

        private fun unregisterContextCleanup(handler: PhantomReference<KZ3Solver>) {
            contextForCleanup.remove(handler)
            handler.clear()
            cleanupStaleContexts()
        }

        private fun cleanupStaleContexts() {
            while (true) {
                val handler = cleanupHandlers.poll() ?: break
                contextForCleanup.remove(handler)?.close()
            }
        }
    }
}
