package org.ksmt.solver.bitwuzla.bindings;

class BitwuzlaException(message: String) : Exception(message) {
    override val message: String?
        get() = "BitwuzlaJNI Exception - " + super.message
}
