package org.ksmt.solver.z3

import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Solver
import com.microsoft.z3.Status
import com.microsoft.z3.Z3Exception
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import org.ksmt.utils.NativeLibraryLoader
import java.lang.ref.PhantomReference
import java.lang.ref.ReferenceQueue
import java.util.IdentityHashMap
import kotlin.time.Duration
import kotlin.time.DurationUnit

open class KZ3Solver(private val ctx: KContext) : KSolver {
    private val z3Ctx = Context()
    private val solver = z3Ctx.mkSolver()
    private val z3InternCtx = KZ3InternalizationContext()
    private var lastCheckStatus = KSolverStatus.UNKNOWN
    private var currentScope: UInt = 0u

    @Suppress("LeakingThis")
    private val contextCleanupActionHandler = registerContextForCleanup(this, z3Ctx)

    private val sortInternalizer by lazy {
        createSortInternalizer(z3InternCtx, z3Ctx)
    }
    private val declInternalizer by lazy {
        createDeclInternalizer(z3InternCtx, z3Ctx, sortInternalizer)
    }
    private val exprInternalizer by lazy {
        createExprInternalizer(z3InternCtx, z3Ctx, sortInternalizer, declInternalizer)
    }
    private val exprConverter by lazy {
        createExprConverter(z3InternCtx, z3Ctx)
    }

    open fun createSortInternalizer(
        internCtx: KZ3InternalizationContext,
        z3Ctx: Context
    ): KZ3SortInternalizer = KZ3SortInternalizer(z3Ctx, internCtx)

    open fun createDeclInternalizer(
        internCtx: KZ3InternalizationContext,
        z3Ctx: Context,
        sortInternalizer: KZ3SortInternalizer
    ): KZ3DeclInternalizer = KZ3DeclInternalizer(z3Ctx, internCtx, sortInternalizer)

    open fun createExprInternalizer(
        internCtx: KZ3InternalizationContext,
        z3Ctx: Context,
        sortInternalizer: KZ3SortInternalizer,
        declInternalizer: KZ3DeclInternalizer
    ): KZ3ExprInternalizer = KZ3ExprInternalizer(ctx, z3Ctx, internCtx, sortInternalizer, declInternalizer)

    open fun createExprConverter(
        internCtx: KZ3InternalizationContext,
        z3Ctx: Context
    ) = KZ3ExprConverter(ctx, internCtx)

    override fun push() {
        solver.push()
        currentScope++
    }

    override fun pop(n: UInt) {
        require(n <= currentScope) {
            "Can not pop $n scope levels because current scope level is $currentScope"
        }
        if (n == 0u) return
        solver.pop(n.toInt())
        currentScope -= n
    }

    override fun assert(expr: KExpr<KBoolSort>) = z3Try {
        val z3Expr = with(exprInternalizer) { expr.internalize() }
        solver.add(z3Expr as BoolExpr)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort> = z3Try {
        val z3Expr = with(exprInternalizer) { expr.internalize() } as BoolExpr
        val trackVar = with(ctx) { boolSort.mkFreshConst("track") }
        val z3TrackVar = with(exprInternalizer) { trackVar.internalize() } as BoolExpr

        solver.assertAndTrack(z3Expr, z3TrackVar)

        trackVar
    }

    override fun check(timeout: Duration): KSolverStatus = z3Try {
        solver.updateTimeout(timeout)
        solver.check().processCheckResult()
    }

    @Suppress("SpreadOperator")
    override fun checkWithAssumptions(
        assumptions: List<KExpr<KBoolSort>>,
        timeout: Duration
    ): KSolverStatus = z3Try {
        val z3Assumptions = with(exprInternalizer) { assumptions.map { it.internalize() as BoolExpr } }
        solver.updateTimeout(timeout)
        solver.check(*z3Assumptions.toTypedArray()).processCheckResult()
    }

    override fun model(): KModel = z3Try {
        require(lastCheckStatus == KSolverStatus.SAT) { "Model are only available after SAT checks" }
        val model = solver.model
        KZ3Model(model, ctx, z3InternCtx, exprInternalizer, exprConverter)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> = z3Try {
        require(lastCheckStatus == KSolverStatus.UNSAT) { "Unsat cores are only available after UNSAT checks" }
        val z3Core = solver.unsatCore
        with(exprConverter) { z3Core.map { it.convert() } }
    }

    override fun reasonOfUnknown(): String = z3Try {
        require(lastCheckStatus == KSolverStatus.UNKNOWN) { "Unknown reason is only available after UNKNOWN checks" }
        solver.reasonUnknown
    }

    override fun close() {
        unregisterContextCleanup(contextCleanupActionHandler)
        z3InternCtx.close()
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
        val params = z3Ctx.mkParams().apply {
            add("timeout", z3Timeout)
        }
        setParameters(params)
    }

    private inline fun <reified T> z3Try(body: () -> T): T = try {
        body()
    } catch (ex: Z3Exception) {
        throw KSolverException(ex)
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

        private val cleanupHandlers = ReferenceQueue<KZ3Solver>()
        private val contextForCleanup = IdentityHashMap<PhantomReference<KZ3Solver>, Context>()

        /** Ensure Z3 native context is closed and all native memory is released.
         * */
        private fun registerContextForCleanup(solver: KZ3Solver, context: Context): PhantomReference<KZ3Solver> {
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
