package org.ksmt.solver

interface KSolverUniversalConfigurationBuilder {
    fun buildBoolParameter(param: String, value: Boolean)
    fun buildIntParameter(param: String, value: Int)
    fun buildStringParameter(param: String, value: String)
    fun buildDoubleParameter(param: String, value: Double)
}
