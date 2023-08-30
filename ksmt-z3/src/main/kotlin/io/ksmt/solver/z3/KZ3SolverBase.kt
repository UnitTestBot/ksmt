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
import io.ksmt.solver.KSolverStatus
import io.ksmt.sort.KBoolSort
import io.ksmt.utils.NativeLibraryLoader
import java.lang.ref.PhantomReference
import java.lang.ref.ReferenceQueue
import java.util.IdentityHashMap
import kotlin.time.Duration
import kotlin.time.DurationUnit

abstract class KZ3SolverBase(protected val ctx: KContext) : KSolver<KZ3SolverConfiguration> {
    protected abstract val z3Ctx: KZ3Context
    protected val solver by lazy { createSolver() }

    protected var lastCheckStatus = KSolverStatus.UNKNOWN
    protected var lastReasonOfUnknown: String? = null
    protected var lastModel: KZ3Model? = null
    protected var lastUnsatCore: List<KExpr<KBoolSort>>? = null

    protected open var currentScope: UInt = 0u

    @Suppress("LeakingThis")
    private val contextCleanupActionHandler = registerContextForCleanup(this, z3Ctx)

    protected val exprInternalizer by lazy {
        createExprInternalizer(z3Ctx)
    }
    protected val exprConverter by lazy {
        createExprConverter(z3Ctx)
    }

    open fun createExprInternalizer(z3Ctx: KZ3Context): KZ3ExprInternalizer = KZ3ExprInternalizer(ctx, z3Ctx)

    open fun createExprConverter(z3Ctx: KZ3Context) = KZ3ExprConverter(ctx, z3Ctx)

    private fun createSolver(): Solver = z3Ctx.z3Try { z3Ctx.nativeContext.mkSolver() }

    override fun configure(configurator: KZ3SolverConfiguration.() -> Unit) = z3Ctx.z3Try {
        val params = z3Ctx.nativeContext.mkParams()
        KZ3SolverConfigurationImpl(params).configurator()
        solver.setParameters(params)
    }

    override fun push(): Unit = z3Ctx.z3Try {
        solver.push()
        z3Ctx.pushAssertionLevel()
        currentScope++
    }

    override fun pop(n: UInt) = z3Ctx.z3Try {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }
        if (n == 0u) return

        solver.pop(n.toInt())
        repeat(n.toInt()) { z3Ctx.popAssertionLevel() }

        currentScope -= n
    }

    override fun assert(expr: KExpr<KBoolSort>) = z3Ctx.z3Try {
        ctx.ensureContextMatch(expr)

        val z3Expr = with(exprInternalizer) { expr.internalizeExpr() }
        solver.solverAssert(z3Expr)

        z3Ctx.assertPendingAxioms(solver)
    }

    protected abstract fun saveTrackedAssertion(track: Long, trackedExpr: KExpr<KBoolSort>)
    protected abstract fun findTrackedExprByTrack(track: Long): KExpr<KBoolSort>?

    override fun assertAndTrack(expr: KExpr<KBoolSort>) = z3Ctx.z3Try {
        ctx.ensureContextMatch(expr)

        val trackExpr = ctx.mkFreshConst("track", ctx.boolSort)
        val z3Expr = with(exprInternalizer) { expr.internalizeExpr() }
        val z3TrackVar = with(exprInternalizer) { trackExpr.internalizeExpr() }

        solver.solverAssertAndTrack(z3Expr, z3TrackVar)
        saveTrackedAssertion(z3TrackVar, expr)
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

    override fun model(): KModel = z3Ctx.z3Try {
        require(lastCheckStatus == KSolverStatus.SAT) {
            "Model are only available after SAT checks, current solver status: $lastCheckStatus"
        }

        val model = lastModel ?: KZ3Model(
            model = solver.model,
            ctx = ctx,
            z3Ctx = z3Ctx,
            internalizer = exprInternalizer
        )
        lastModel = model

        model
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = z3Ctx.z3Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }

        val unsatCore = lastUnsatCore ?: with(exprConverter) {
            val solverUnsatCore = solver.solverGetUnsatCore()
            solverUnsatCore.map { solverUnsatCoreExpr ->
                findTrackedExprByTrack(solverUnsatCoreExpr) ?: solverUnsatCoreExpr.convertExpr()
            }
        }
        lastUnsatCore = unsatCore

        unsatCore
    }

    override fun reasonOfUnknown(): String = z3Ctx.z3Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) { "Unknown reason is only available after UNKNOWN checks" }
        lastReasonOfUnknown ?: solver.reasonUnknown
    }

    override fun interrupt() = z3Ctx.z3Try {
        solver.interrupt()
    }

    override fun close() {
        unregisterContextCleanup(contextCleanupActionHandler)
        z3Ctx.close()
    }

    protected fun Status?.processCheckResult() = when (this) {
        Status.SATISFIABLE -> KSolverStatus.SAT
        Status.UNSATISFIABLE -> KSolverStatus.UNSAT
        Status.UNKNOWN -> KSolverStatus.UNKNOWN
        null -> KSolverStatus.UNKNOWN
    }.also { lastCheckStatus = it }

    protected fun Solver.updateTimeout(timeout: Duration) {
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

    protected fun invalidateSolverState() {
        lastReasonOfUnknown = null
        lastCheckStatus = KSolverStatus.UNKNOWN
        lastModel = null
        lastUnsatCore = null
    }

    protected inline fun z3TryCheck(body: () -> KSolverStatus): KSolverStatus = try {
        invalidateSolverState()
        body()
    } catch (ex: Z3Exception) {
        lastReasonOfUnknown = ex.message
        KSolverStatus.UNKNOWN.also { lastCheckStatus = it }
    }

    companion object {
        init {
            System.setProperty("z3.skipLibraryLoad", "true")
            NativeLibraryLoader.load { os ->
                when (os) {
                    NativeLibraryLoader.OS.LINUX -> listOf("libz3", "libz3java")
                    NativeLibraryLoader.OS.MACOS -> listOf("libz3", "libz3java")
                    NativeLibraryLoader.OS.WINDOWS -> listOf("vcruntime140", "vcruntime140_1", "libz3", "libz3java")
                }
            }
        }

        private val cleanupHandlers = ReferenceQueue<KSolver<KZ3SolverConfiguration>>()
        private val contextForCleanup = IdentityHashMap<PhantomReference<KSolver<KZ3SolverConfiguration>>, KZ3Context>()

        /** Ensure Z3 native context is closed and all native memory is released.
         * */
        private fun registerContextForCleanup(
            solver: KSolver<KZ3SolverConfiguration>,
            context: KZ3Context
        ): PhantomReference<KSolver<KZ3SolverConfiguration>> {
            cleanupStaleContexts()
            val cleanupHandler = PhantomReference(solver, cleanupHandlers)
            contextForCleanup[cleanupHandler] = context

            return cleanupHandler
        }

        private fun unregisterContextCleanup(handler: PhantomReference<KSolver<KZ3SolverConfiguration>>) {
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
