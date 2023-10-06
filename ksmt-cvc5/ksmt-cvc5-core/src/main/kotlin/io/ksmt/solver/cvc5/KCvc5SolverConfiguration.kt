package io.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KSolverUnsupportedParameterException

interface KCvc5SolverConfiguration : KSolverConfiguration {
    fun setCvc5Option(option: String, value: String)

    fun setCvc5Logic(value: String)

    override fun setStringParameter(param: String, value: String) {
        if (param == LOGIC_PARAM_NAME) {
            setCvc5Logic(value)
        } else {
            setCvc5Option(param, value)
        }
    }

    override fun setBoolParameter(param: String, value: Boolean) {
        throw KSolverUnsupportedParameterException("Cvc5 does not options with boolean value")
    }

    override fun setIntParameter(param: String, value: Int) {
        throw KSolverUnsupportedParameterException("Cvc5 does not options with int value")
    }

    override fun setDoubleParameter(param: String, value: Double) {
        throw KSolverUnsupportedParameterException("Cvc5 does not options with double value")
    }

    companion object {
        const val LOGIC_PARAM_NAME = "logic"
    }
}

class KCvc5SolverConfigurationImpl(val solver: Solver) : KCvc5SolverConfiguration {
    override fun setCvc5Option(option: String, value: String) {
        solver.setOption(option, value)
    }

    override fun setCvc5Logic(value: String) {
        solver.setLogic(value)
    }
}

class KCvc5SolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KCvc5SolverConfiguration {
    override fun setCvc5Option(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }

    override fun setCvc5Logic(value: String) {
        builder.buildStringParameter(KCvc5SolverConfiguration.LOGIC_PARAM_NAME, value)
    }
}
