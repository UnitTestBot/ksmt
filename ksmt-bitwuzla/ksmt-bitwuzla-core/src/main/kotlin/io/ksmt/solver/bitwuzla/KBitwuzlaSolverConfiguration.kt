package io.ksmt.solver.bitwuzla

import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KSolverUnsupportedParameterException
import org.ksmt.solver.bitwuzla.bindings.Bitwuzla
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.bitwuzla.bindings.Native

interface KBitwuzlaSolverConfiguration : KSolverConfiguration {
    fun setBitwuzlaOption(option: BitwuzlaOption, value: Int)
    fun setBitwuzlaOption(option: BitwuzlaOption, value: String)

    override fun setIntParameter(param: String, value: Int) {
        val option = BitwuzlaOption.forName(param)
            ?: throw KSolverUnsupportedParameterException("Int parameter $param is not supported in Bitwuzla")

        setBitwuzlaOption(option, value)
    }

    override fun setStringParameter(param: String, value: String) {
        val option = BitwuzlaOption.forName(param)
            ?: throw KSolverUnsupportedParameterException("String parameter $param is not supported in Bitwuzla")

        setBitwuzlaOption(option, value)
    }

    override fun setBoolParameter(param: String, value: Boolean) {
        throw KSolverUnsupportedParameterException("Boolean parameter $param is not supported in Bitwuzla")
    }

    override fun setDoubleParameter(param: String, value: Double) {
        throw KSolverUnsupportedParameterException("Double parameter $param is not supported in Bitwuzla")
    }
}

class KBitwuzlaSolverConfigurationImpl(private val bitwuzla: Bitwuzla) : KBitwuzlaSolverConfiguration {
    override fun setBitwuzlaOption(option: BitwuzlaOption, value: Int) {
        Native.bitwuzlaSetOption(bitwuzla, option, value)
    }

    override fun setBitwuzlaOption(option: BitwuzlaOption, value: String) {
        Native.bitwuzlaSetOptionStr(bitwuzla, option, value)
    }
}

class KBitwuzlaSolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KBitwuzlaSolverConfiguration {
    override fun setBitwuzlaOption(option: BitwuzlaOption, value: Int) {
        builder.buildIntParameter(option.name, value)
    }

    override fun setBitwuzlaOption(option: BitwuzlaOption, value: String) {
        builder.buildStringParameter(option.name, value)
    }
}
