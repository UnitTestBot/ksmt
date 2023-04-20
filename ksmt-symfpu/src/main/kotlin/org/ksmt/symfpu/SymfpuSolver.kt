package org.ksmt.symfpu

import org.ksmt.KContext
import org.ksmt.decl.KConstDecl
import org.ksmt.decl.KDecl
import org.ksmt.expr.KExpr
import org.ksmt.expr.KUninterpretedSortValue
import org.ksmt.solver.KModel
import org.ksmt.solver.KSolver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverStatus
import org.ksmt.sort.KBoolSort
import org.ksmt.sort.KBvSort
import org.ksmt.sort.KFpSort
import org.ksmt.sort.KSort
import org.ksmt.sort.KUninterpretedSort
import org.ksmt.utils.cast
import kotlin.time.Duration

open class SymfpuSolver<Config : KSolverConfiguration>(val solver: KSolver<Config>,
                                                       val ctx: KContext) : KSolver<Config> {


    private val transformer = FpToBvTransformer(ctx)

    override fun configure(configurator: Config.() -> Unit) {
        solver.configure(configurator)
    }

    override fun assert(expr: KExpr<KBoolSort>) = solver.assert(transformer.applyAndGetExpr(expr))

    override fun assertAndTrack(expr: KExpr<KBoolSort>, trackVar: KConstDecl<KBoolSort>) =
        solver.assertAndTrack(transformer.applyAndGetExpr(expr), trackVar)

    override fun push() = solver.push()

    override fun pop(n: UInt) = solver.pop(n)


    override fun check(timeout: Duration): KSolverStatus = solver.check(timeout)

    override fun checkWithAssumptions(assumptions: List<KExpr<KBoolSort>>, timeout: Duration): KSolverStatus =
        solver.checkWithAssumptions(assumptions.map(transformer::applyAndGetExpr), timeout)

    override fun model(): KModel = Model(solver.model())

    override fun unsatCore(): List<KExpr<KBoolSort>> = solver.unsatCore()

    override fun reasonOfUnknown(): String = solver.reasonOfUnknown()

    override fun interrupt() = solver.interrupt()

    override fun close() = solver.close()

    inner class Model(val kModel: KModel) : KModel {
        override val declarations: Set<KDecl<*>>
            get() = kModel.declarations

        override val uninterpretedSorts: Set<KUninterpretedSort>
            get() = kModel.uninterpretedSorts

        override fun <T : KSort> eval(expr: KExpr<T>, isComplete: Boolean): KExpr<T> = with(expr.ctx) {
            val eval = kModel.eval(transformer.applyAndGetExpr(expr), isComplete)
            val sort = expr.sort
            if (sort is KFpSort) {
                val bv: KExpr<KBvSort> = eval.cast()
                return unpackBiased(sort, bv).toFp().cast()
            } else eval
        }

        override fun <T : KSort> interpretation(decl: KDecl<T>): KModel.KFuncInterp<T>? {
            return kModel.interpretation(decl)
        }

        override fun uninterpretedSortUniverse(sort: KUninterpretedSort): Set<KUninterpretedSortValue>? {
            return kModel.uninterpretedSortUniverse(sort)
        }

        override fun detach(): KModel {
            return kModel.detach()
        }
    }
}
