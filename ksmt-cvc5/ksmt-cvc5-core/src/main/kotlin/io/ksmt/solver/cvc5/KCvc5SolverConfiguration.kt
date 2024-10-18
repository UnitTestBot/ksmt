package io.ksmt.solver.cvc5

import io.github.cvc5.Solver
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KSolverUnsupportedParameterException
import io.ksmt.solver.KTheory
import io.ksmt.solver.smtLib2String

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

class KCvc5SolverLazyConfiguration : KCvc5SolverConfiguration {
    private var logicConfiguration: String? = null
    private val options = mutableMapOf<String, String>()

    override fun setCvc5Option(option: String, value: String) {
        options[option] = value
    }

    override fun setCvc5Logic(value: String) {
        logicConfiguration = value
    }

    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        logicConfiguration = theories.smtLib2String(quantifiersAllowed)
    }

    fun configure(solver: Solver) {
        logicConfiguration?.let { solver.setLogic(it) }
        options.forEach { (option, value) -> solver.setOption(option, value) }
    }
}

class KCvc5SolverOptionsConfiguration(val solver: Solver) : KCvc5SolverConfiguration {
    override fun setCvc5Option(option: String, value: String) {
        solver.setOption(option, value)
    }

    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        throw KSolverException("Solver logic already configured")
    }

    override fun setCvc5Logic(value: String) {
        throw KSolverException("Solver logic already configured")
    }
}

class KCvc5SolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KCvc5SolverConfiguration {
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        builder.buildOptimizeForTheories(theories, quantifiersAllowed)
    }

    override fun setCvc5Option(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }

    override fun setCvc5Logic(value: String) {
        builder.buildStringParameter(KCvc5SolverConfiguration.LOGIC_PARAM_NAME, value)
    }
}
