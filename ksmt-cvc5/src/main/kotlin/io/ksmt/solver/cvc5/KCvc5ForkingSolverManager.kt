package io.ksmt.solver.cvc5

import io.ksmt.KContext
import io.ksmt.solver.KForkingSolver
import io.ksmt.solver.KForkingSolverManager
import java.util.concurrent.ConcurrentHashMap

open class KCvc5ForkingSolverManager(private val ctx: KContext) : KForkingSolverManager<KCvc5SolverConfiguration> {

    private val solvers: MutableSet<KCvc5ForkingSolver> = ConcurrentHashMap.newKeySet()

    override fun mkForkingSolver(): KForkingSolver<KCvc5SolverConfiguration> {
        return KCvc5ForkingSolver(ctx, this, null).also { solvers += it }
    }

    internal fun mkForkingSolver(parent: KCvc5ForkingSolver): KForkingSolver<KCvc5SolverConfiguration> {
        return KCvc5ForkingSolver(ctx, this, parent).also { solvers += it }
    }

    /**
     * unregister [solver] for this manager
     */
    internal fun close(solver: KCvc5ForkingSolver) {
        solvers -= solver
    }

    override fun close() {
        solvers.forEach(KCvc5ForkingSolver::close)
    }
}
