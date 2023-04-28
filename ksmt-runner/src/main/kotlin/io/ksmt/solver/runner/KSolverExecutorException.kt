package io.ksmt.solver.runner

import io.ksmt.runner.core.WorkerInitializationFailedException

sealed class KSolverExecutorException : Exception {
    constructor(message: String?) : super(message)
    constructor(cause: Throwable?) : super(cause)
    override fun fillInStackTrace(): Throwable = this
}

class KSolverExecutorNotAliveException : KSolverExecutorException("Solver executor is not alive")
class KSolverExecutorTimeoutException(message: String?) : KSolverExecutorException(message)
class KSolverExecutorWorkerInitializationException(
    reason: WorkerInitializationFailedException
): KSolverExecutorException(reason)
class KSolverExecutorOtherException(cause: Throwable?) : KSolverExecutorException(cause)
