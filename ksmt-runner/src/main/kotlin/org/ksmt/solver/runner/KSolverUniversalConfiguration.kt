package org.ksmt.solver.runner

import org.ksmt.runner.models.generated.ConfigurationParamKind
import org.ksmt.runner.models.generated.SolverConfigurationParam
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverUniversalConfigurationBuilder

class KSolverRunnerUniversalConfigurator : KSolverUniversalConfigurationBuilder {
    val config = mutableListOf<SolverConfigurationParam>()

    override fun buildBoolParameter(param: String, value: Boolean) {
        config += SolverConfigurationParam(ConfigurationParamKind.Bool, param, "$value")
    }

    override fun buildIntParameter(param: String, value: Int) {
        config += SolverConfigurationParam(ConfigurationParamKind.Int, param, "$value")
    }

    override fun buildDoubleParameter(param: String, value: Double) {
        config += SolverConfigurationParam(ConfigurationParamKind.Double, param, "$value")
    }

    override fun buildStringParameter(param: String, value: String) {
        config += SolverConfigurationParam(ConfigurationParamKind.String, param, value)
    }
}

fun KSolverConfiguration.addUniversalParam(param: SolverConfigurationParam): Unit = with(param) {
    when (kind) {
        ConfigurationParamKind.String -> setStringParameter(name, value)
        ConfigurationParamKind.Bool -> setBoolParameter(name, value.toBooleanStrict())
        ConfigurationParamKind.Int -> setIntParameter(name, value.toInt())
        ConfigurationParamKind.Double -> setDoubleParameter(name, value.toDouble())
    }
}
