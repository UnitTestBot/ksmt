package io.ksmt.solver.yices

import com.sri.yices.Config
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KSolverUnsupportedParameterException

interface KYicesSolverConfiguration : KSolverConfiguration {
    fun setYicesOption(option: String, value: String)

    override fun setStringParameter(param: String, value: String) {
        setYicesOption(param, value)
    }

    override fun setBoolParameter(param: String, value: Boolean) {
        throw KSolverUnsupportedParameterException("Boolean parameter $param is no supported in Yices")
    }

    override fun setIntParameter(param: String, value: Int) {
        throw KSolverUnsupportedParameterException("Int parameter $param is no supported in Yices")
    }

    override fun setDoubleParameter(param: String, value: Double) {
        throw KSolverUnsupportedParameterException("Double parameter $param is no supported in Yices")
    }
}

class KYicesSolverConfigurationImpl(private val config: Config) : KYicesSolverConfiguration {
    override fun setYicesOption(option: String, value: String) {
        config.set(option, value)
    }
}

class KYicesSolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KYicesSolverConfiguration {
    override fun setYicesOption(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }
}
