package org.ksmt.solver

@Suppress("unused")
open class KSolverException : Exception {
    constructor() : super()
    constructor(cause: Throwable) : super(cause)
    constructor(message: String) : super(message)
    constructor(message: String, cause: Throwable) : super(message, cause)
}
