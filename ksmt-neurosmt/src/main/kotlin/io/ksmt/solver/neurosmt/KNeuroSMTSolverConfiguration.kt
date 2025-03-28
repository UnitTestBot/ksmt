package io.ksmt.solver.neurosmt

import io.ksmt.solver.KSolverConfiguration

interface KNeuroSMTSolverConfiguration : KSolverConfiguration {
    fun setOption(option: String, value: Boolean)
    fun setOption(option: String, value: Int)
    fun setOption(option: String, value: Double)
    fun setOption(option: String, value: String)

    override fun setBoolParameter(param: String, value: Boolean) {
        setOption(param, value)
    }

    override fun setIntParameter(param: String, value: Int) {
        setOption(param, value)
    }

    override fun setDoubleParameter(param: String, value: Double) {
        setOption(param, value)
    }

    override fun setStringParameter(param: String, value: String) {
        setOption(param, value)
    }
}

class KNeuroSMTSolverConfigurationImpl(private val params: Any?) : KNeuroSMTSolverConfiguration {
    override fun setOption(option: String, value: Boolean) {
        TODO("Not yet implemented")
    }

    override fun setOption(option: String, value: Int) {
        TODO("Not yet implemented")
    }

    override fun setOption(option: String, value: Double) {
        TODO("Not yet implemented")
    }

    override fun setOption(option: String, value: String) {
        TODO("Not yet implemented")
    }
}