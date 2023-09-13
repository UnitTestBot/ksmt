package io.ksmt.solver.z3

import com.microsoft.z3.Params
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverUniversalConfigurationBuilder

interface KZ3SolverConfiguration : KSolverConfiguration {
    fun setZ3Option(option: String, value: Boolean)
    fun setZ3Option(option: String, value: Int)
    fun setZ3Option(option: String, value: Double)
    fun setZ3Option(option: String, value: String)

    override fun setBoolParameter(param: String, value: Boolean) {
        setZ3Option(param, value)
    }

    override fun setIntParameter(param: String, value: Int) {
        setZ3Option(param, value)
    }

    override fun setStringParameter(param: String, value: String) {
        setZ3Option(param, value)
    }

    override fun setDoubleParameter(param: String, value: Double) {
        setZ3Option(param, value)
    }
}

class KZ3SolverConfigurationImpl(private val params: Params) : KZ3SolverConfiguration {
    override fun setZ3Option(option: String, value: Boolean) {
        params.add(option, value)
    }

    override fun setZ3Option(option: String, value: Int) {
        params.add(option, value)
    }

    override fun setZ3Option(option: String, value: Double) {
        params.add(option, value)
    }

    override fun setZ3Option(option: String, value: String) {
        params.add(option, value)
    }
}

class KZ3SolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KZ3SolverConfiguration {
    override fun setZ3Option(option: String, value: Boolean) {
        builder.buildBoolParameter(option, value)
    }

    override fun setZ3Option(option: String, value: Int) {
        builder.buildIntParameter(option, value)
    }

    override fun setZ3Option(option: String, value: Double) {
        builder.buildDoubleParameter(option, value)
    }

    override fun setZ3Option(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }
}
