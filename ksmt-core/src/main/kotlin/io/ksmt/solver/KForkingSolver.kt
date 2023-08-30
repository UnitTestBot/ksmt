package io.ksmt.solver

/**
 * A solver capable of creating forks (copies) of itself, preserving assertions and assertion scopes
 *
 * @see KForkingSolverManager
 */
interface KForkingSolver<Config : KSolverConfiguration> : KSolver<Config> {

    /**
     * Creates forked solver
     */
    fun fork(): KForkingSolver<Config>
}
