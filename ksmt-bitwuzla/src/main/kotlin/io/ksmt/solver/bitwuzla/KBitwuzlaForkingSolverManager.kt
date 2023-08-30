package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import java.util.concurrent.ConcurrentHashMap

/**
 * Responsible for creation and managing of [KBitwuzlaForkingSolver].
 *
 * Neither native cache is shared between [KBitwuzlaForkingSolver]s
 * because cache sharing is not supported in bitwuzla.
 */
class KBitwuzlaForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KBitwuzlaSolverConfiguration> {
    private val solvers = ConcurrentHashMap.newKeySet<KBitwuzlaForkingSolver>()

    override fun createForkingSolver(): KForkingSolver<KBitwuzlaSolverConfiguration> {
        return KBitwuzlaForkingSolver(ctx, this, null).also {
            solvers += it
        }
    }

    internal fun createForkingSolver(parent: KBitwuzlaForkingSolver) = KBitwuzlaForkingSolver(ctx, this, parent)
        .also { solvers += it }

    internal fun close(solver: KBitwuzlaForkingSolver) {
        solvers -= solver
    }

    override fun close() {
        solvers.forEach(KBitwuzlaForkingSolver::close)
    }
}
