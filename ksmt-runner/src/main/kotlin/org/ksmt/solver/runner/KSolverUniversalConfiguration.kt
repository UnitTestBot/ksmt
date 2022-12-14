package org.ksmt.solver.runner

import org.ksmt.runner.models.generated.ConfigurationParamKind
import org.ksmt.runner.models.generated.SolverConfigurationParam
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.bitwuzla.KBitwuzlaSolverConfiguration
import org.ksmt.solver.bitwuzla.bindings.BitwuzlaOption
import org.ksmt.solver.z3.KZ3SolverConfiguration

interface KSolverUniversalConfigurationBuilder<Config : KSolverConfiguration> {
    fun build(body: Config.() -> Unit): List<SolverConfigurationParam>
}

class KZ3SolverUniversalConfigurationBuilder : KSolverUniversalConfigurationBuilder<KZ3SolverConfiguration> {
    override fun build(body: KZ3SolverConfiguration.() -> Unit): List<SolverConfigurationParam> =
        buildList { KZ3SolverUniversalConfiguration(this).body() }
}

class KBitwuzlaSolverUniversalConfigurationBuilder :
    KSolverUniversalConfigurationBuilder<KBitwuzlaSolverConfiguration> {
    override fun build(body: KBitwuzlaSolverConfiguration.() -> Unit): List<SolverConfigurationParam> =
        buildList { KBitwuzlaSolverUniversalConfiguration(this).body() }
}

private class KZ3SolverUniversalConfiguration(
    private val config: MutableList<SolverConfigurationParam>
) : KZ3SolverConfiguration {
    override fun setZ3Option(option: String, value: Boolean) {
        config += SolverConfigurationParam(ConfigurationParamKind.Bool, option, "$value")
    }

    override fun setZ3Option(option: String, value: Int) {
        config += SolverConfigurationParam(ConfigurationParamKind.Int, option, "$value")
    }

    override fun setZ3Option(option: String, value: Double) {
        config += SolverConfigurationParam(ConfigurationParamKind.Double, option, "$value")
    }

    override fun setZ3Option(option: String, value: String) {
        config += SolverConfigurationParam(ConfigurationParamKind.String, option, value)
    }
}

private class KBitwuzlaSolverUniversalConfiguration(
    private val config: MutableList<SolverConfigurationParam>
) : KBitwuzlaSolverConfiguration {
    override fun setBitwuzlaOption(option: BitwuzlaOption, value: Int) {
        config += SolverConfigurationParam(ConfigurationParamKind.Int, option.name, "$value")
    }

    override fun setBitwuzlaOption(option: BitwuzlaOption, value: String) {
        config += SolverConfigurationParam(ConfigurationParamKind.String, option.name, value)
    }
}

fun KZ3SolverConfiguration.addUniversalParam(param: SolverConfigurationParam): Unit = with(param) {
    when (kind) {
        ConfigurationParamKind.String -> setZ3Option(name, value)
        ConfigurationParamKind.Bool -> setZ3Option(name, value.toBooleanStrict())
        ConfigurationParamKind.Int -> setZ3Option(name, value.toInt())
        ConfigurationParamKind.Double -> setZ3Option(name, value.toDouble())
    }
}

fun KBitwuzlaSolverConfiguration.addUniversalParam(param: SolverConfigurationParam): Unit = with(param) {
    val option = BitwuzlaOption.valueOf(name)
    when (kind) {
        ConfigurationParamKind.String -> setBitwuzlaOption(option, value)
        ConfigurationParamKind.Int -> setBitwuzlaOption(option, value.toInt())
        else -> error("Unsupported option kind: $kind")
    }
}
