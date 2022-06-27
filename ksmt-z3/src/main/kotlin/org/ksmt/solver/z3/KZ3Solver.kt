package org.ksmt.solver.z3

import com.microsoft.z3.BoolExpr
import com.microsoft.z3.Context
import com.microsoft.z3.Status
import com.microsoft.z3.Z3Exception
import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverException
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort

open class KZ3Solver(val ctx: KContext) : KSolver {
    private val z3Ctx = Context()
    private val solver = z3Ctx.mkSolver()
    private val z3InternCtx = KZ3InternalizationContext()
    private var lastCheckStatus = KSolverStatus.UNKNOWN
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

    override fun assert(expr: KExpr<KBoolSort>) = try {
        val z3Expr = with(exprInternalizer) { expr.internalize() }
        solver.add(z3Expr as BoolExpr)
    } catch (ex: Z3Exception) {
        throw KSolverException(ex)
    }

    override fun check(): KSolverStatus = try {
        val status = solver.check() ?: Status.UNKNOWN
        when (status) {
            Status.SATISFIABLE -> KSolverStatus.SAT
            Status.UNSATISFIABLE -> KSolverStatus.UNSAT
            Status.UNKNOWN -> KSolverStatus.UNKNOWN
        }.also { lastCheckStatus = it }
    } catch (ex: Z3Exception) {
        throw KSolverException(ex)
    }

    override fun model(): KModel = try {
        require(lastCheckStatus == KSolverStatus.SAT) { "Model are only available after SAT checks" }
        val model = solver.model
        KZ3Model(model, ctx, z3InternCtx, exprInternalizer, z3Ctx, exprConverter)
    } catch (ex: Z3Exception) {
        throw KSolverException(ex)
    }

    override fun close() {
        z3InternCtx.close()
        z3Ctx.close()
    }
}
