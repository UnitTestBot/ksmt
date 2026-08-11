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

/**
 * Sort of the descriptor values ksmt uses to keep uninterpreted sort values distinct
 * (see `KCvc5ExprInternalizer.transform(KUninterpretedSortValue)`).
 * */
enum class KCvc5UninterpretedValueDescriptor { INT, BV }

class KCvc5SolverLazyConfiguration : KCvc5SolverConfiguration {
    private var logicConfiguration: String? = null
    private val options = mutableMapOf<String, String>()

    /**
     * Theories the caller declared, or `null` when they did not, in which case
     * nothing is known about the query and every theory has to be assumed.
     * */
    var declaredTheories: Set<KTheory>? = null
        private set

    var valueDescriptor: KCvc5UninterpretedValueDescriptor = KCvc5UninterpretedValueDescriptor.INT
        private set

    override fun setCvc5Option(option: String, value: String) {
        options[option] = value
    }

    override fun setCvc5Logic(value: String) {
        logicConfiguration = value
    }

    /**
     * ksmt keeps uninterpreted sort values distinct through values of a descriptor sort, so every
     * query constrains that sort whether or not the caller declared its theory. A logic that
     * forbids it makes cvc5 reject the check, so the two are chosen together.
     * */
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        declaredTheories = theories

        if (theories.isNullOrEmpty()) {
            logicConfiguration = theories.smtLib2String(quantifiersAllowed)
            return
        }

        val hasIntegerArithmetic = KTheory.LIA in theories || KTheory.NIA in theories

        if (KTheory.BV in theories && !hasIntegerArithmetic) {
            valueDescriptor = KCvc5UninterpretedValueDescriptor.BV
            logicConfiguration = theories.smtLib2String(quantifiersAllowed)
            return
        }

        val theoriesWithDescriptor = if (hasIntegerArithmetic) theories else theories + KTheory.LIA
        logicConfiguration = theoriesWithDescriptor.smtLib2String(quantifiersAllowed)
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
