package org.ksmt.solver.cvc5

import io.github.cvc5.Solver
import org.ksmt.solver.KSolverConfiguration
import org.ksmt.solver.KSolverUniversalConfigurationBuilder
import org.ksmt.solver.KSolverUnsupportedParameterException

interface KCvc5SolverConfiguration : KSolverConfiguration {
    fun setCvc5Option(option: String, value: String)

    override fun setStringParameter(param: String, value: String) {
        setCvc5Option(param, value)
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
}

class KCvc5SolverConfigurationImpl(val solver: Solver) : KCvc5SolverConfiguration {
    override fun setCvc5Option(option: String, value: String) {
        solver.setOption(option, value)
    }
}

class KCvc5SolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KCvc5SolverConfiguration {
    override fun setCvc5Option(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }
}
