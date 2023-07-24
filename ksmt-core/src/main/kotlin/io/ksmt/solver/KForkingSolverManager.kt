package io.ksmt.solver

/**
 * Responsible for creation of [KForkingSolver] and managing its lifetime
 *
 * @see KForkingSolver
 */
interface KForkingSolverManager <Config : KSolverConfiguration> : AutoCloseable {

    fun mkForkingSolver(): KForkingSolver<Config>

    /**
     * Closes the manager and all opened solvers ([KForkingSolver]) managed by this
     */
    override fun close()
}
