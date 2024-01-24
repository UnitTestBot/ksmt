package io.ksmt.solver.maxsmt.solvers.exceptions

internal class NotYetImplementedException : RuntimeException {
    constructor() : super()
    constructor(message: String) : super(message)
}
