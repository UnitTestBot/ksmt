package io.ksmt.solver.yices

import io.ksmt.KContext

/**
 * Yices Context that allows forking and resources sharing via [KYicesForkingSolverManager].
 * To track resources, we have to use a unique solver specific ID, related to them.
 * @param [solver] is used as "specific ID" for resource tracking,
 * because we can't use initialized [com.sri.yices.Context] here.
 */
class KYicesForkingContext(
    ctx: KContext,
    manager: KYicesForkingSolverManager,
    solver: KYicesForkingSolver
) : KYicesContext(ctx) {
    override val expressions = manager.findExpressionsCache(solver)
    override val yicesExpressions = manager.findExpressionsReversedCache(solver)

    override val sorts = manager.findSortsCache(solver)
    override val yicesSorts = manager.findSortsReversedCache(solver)

    override val decls = manager.findDeclsCache(solver)
    override val yicesDecls = manager.findDeclsReversedCache(solver)

    override val vars = manager.findVarsCache(solver)
    override val yicesVars = manager.findVarsReversedCache(solver)

    override val yicesTypes = manager.findTypesCache(solver)
    override val yicesTerms = manager.findTermsCache(solver)

    private val maxValueIndexAtomic = manager.findMaxUninterpretedSortValueIdx(solver)

    override var maxValueIndex: Int
        get() = maxValueIndexAtomic.get()
        set(value) {
            maxValueIndexAtomic.set(value)
        }

    override val uninterpretedSortValuesTracker = manager.createUninterpretedValuesTracker(solver)

    override fun close() {
        if (isClosed) return
        isClosed = true

        performGc()
    }
}
