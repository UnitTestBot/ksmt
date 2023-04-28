package io.ksmt.solver

interface KSolverConfiguration {
    fun setBoolParameter(param: String, value: Boolean)
    fun setIntParameter(param: String, value: Int)
    fun setStringParameter(param: String, value: String)
    fun setDoubleParameter(param: String, value: Double)
}
