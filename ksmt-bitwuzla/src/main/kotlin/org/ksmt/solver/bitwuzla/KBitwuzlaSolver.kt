package org.ksmt.solver.bitwuzla

import org.ksmt.KContext
import org.ksmt.expr.KExpr
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverStatus
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaResult
import org.ksmt.solver.bitwuzla.bindings.Native
import org.ksmt.sort.KBoolSort
import kotlin.time.Duration

open class KBitwuzlaSolver(private val ctx: KContext) : KSolver {
    open val bitwuzlaCtx = KBitwuzlaContext()
    open val exprInternalizer: KBitwuzlaExprInternalizer by lazy {
        KBitwuzlaExprInternalizer(ctx, bitwuzlaCtx)
    }
    open val exprConverter: KBitwuzlaExprConverter by lazy {
        KBitwuzlaExprConverter(ctx, bitwuzlaCtx)
    }
    private var lastCheckStatus = KSolverStatus.UNKNOWN

    init {
        Native.bitwuzlaSetOption(bitwuzlaCtx.bitwuzla, BitwuzlaOption.BITWUZLA_OPT_PRODUCE_MODELS, 1)
    }

    override fun assert(expr: KExpr<KBoolSort>) {
        val term = with(exprInternalizer) { expr.internalize() }
        bitwuzlaCtx.assert(term)
    }

    override fun assertAndTrack(expr: KExpr<KBoolSort>): KExpr<KBoolSort> {
        TODO("Not yet implemented")
    }

    override fun push() {
        TODO("Not yet implemented")
    }

    override fun pop(n: UInt) {
        TODO("Not yet implemented")
    }

    override fun check(timeout: Duration): KSolverStatus {
        val status = bitwuzlaCtx.check()
        return when (status) {
            BitwuzlaResult.BITWUZLA_SAT -> KSolverStatus.SAT
            BitwuzlaResult.BITWUZLA_UNSAT -> KSolverStatus.UNSAT
            BitwuzlaResult.BITWUZLA_UNKNOWN -> KSolverStatus.UNKNOWN
        }.also { lastCheckStatus = it }
    }

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus {
        TODO("Not yet implemented")
    }

    override fun model(): KModel {
        require(lastCheckStatus == KSolverStatus.SAT) { "Model are only available after SAT checks" }
        return KBitwuzlaModel(ctx, bitwuzlaCtx, exprInternalizer, exprConverter)
    }

    override fun unsatCore(): List<KExpr<KBoolSort>> {
        TODO("Not yet implemented")
    }

    override fun reasonOfUnknown(): String {
        TODO("Not yet implemented")
    }

    override fun close() {
        bitwuzlaCtx.close()
    }
}
