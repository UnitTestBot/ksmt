package io.ksmt.solver

@Suppress("OVERLOADS_INTERFACE", "INAPPLICABLE_JVM_NAME")
interface KSolverConfiguration {
    fun setBoolParameter(param: String, value: Boolean)
    fun setIntParameter(param: String, value: Int)
    fun setStringParameter(param: String, value: String)
    fun setDoubleParameter(param: String, value: Double)

    /**
     * Specialize the solver to work with the provided theories.
     *
     * [theories] a set of theories.
     * If the provided theories are null, the solver is specialized to work with all supported theories.
     * If the provided theory set is empty, the solver is configured to work only with propositional formulas.
     *
     * [quantifiersAllowed] allows or disallows formulas with quantifiers.
     * If quantifiers are not allowed, the solver is specialized to work with Quantifier Free formulas.
     * * */
    @JvmOverloads
    @JvmName("optimizeForTheories")
    fun optimizeForTheories(theories: Set<KTheory>? = null, quantifiersAllowed: Boolean = false)
}
