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
    override val expressions = manager.getExpressionsCache(solver)
    override val yicesExpressions = manager.getExpressionsReversedCache(solver)

    override val sorts = manager.getSortsCache(solver)
    override val yicesSorts = manager.getSortsReversedCache(solver)

    override val decls = manager.getDeclsCache(solver)
    override val yicesDecls = manager.getDeclsReversedCache(solver)

    override val vars = manager.getVarsCache(solver)
    override val yicesVars = manager.getVarsReversedCache(solver)

    override val yicesTypes = manager.getTypesCache(solver)
    override val yicesTerms = manager.getTermsCache(solver)

    private val maxValueIndexAtomic = manager.getMaxUninterpretedSortValueIdx(solver)

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
