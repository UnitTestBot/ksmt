package io.ksmt.solver.bitwuzla

import io.ksmt.KContext
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import java.util.concurrent.ConcurrentHashMap

class KBitwuzlaForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KBitwuzlaSolverConfiguration> {
    private val solvers = ConcurrentHashMap.newKeySet<KBitwuzlaForkingSolver>()

    override fun mkForkingSolver(): KForkingSolver<KBitwuzlaSolverConfiguration> {
        return KBitwuzlaForkingSolver(ctx, this, null).also {
            solvers += it
        }
    }

    internal fun mkForkingSolver(parent: KBitwuzlaForkingSolver) = KBitwuzlaForkingSolver(ctx, this, parent).also {
        solvers += it
    }

    internal fun close(solver: KBitwuzlaForkingSolver) {
        solvers -= solver
    }

    override fun close() {
        solvers.forEach(KBitwuzlaForkingSolver::close)
    }
}
