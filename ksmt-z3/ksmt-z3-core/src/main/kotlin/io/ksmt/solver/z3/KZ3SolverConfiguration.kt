package io.ksmt.solver.z3

import com.microsoft.z3.Params
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.smtLib2String

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

sealed class KZ3SolverConfigurationImpl(val params: Params) : KZ3SolverConfiguration {
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

class KZ3SolverLazyConfiguration(params: Params) : KZ3SolverConfigurationImpl(params) {
    var logicConfiguration: String? = null
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        if (theories.isNullOrEmpty() || !supportedLogicCombination(theories)) {
            logicConfiguration = null
            return
        }

        logicConfiguration = theories.smtLib2String(quantifiersAllowed)
    }

    /**
     * Z3 doesn't provide special solver for the following theory combinations
     * */
    private fun supportedLogicCombination(theories: Set<KTheory>): Boolean {
        if (BV in theories) {
            if (setOf(LIA, LRA, NIA, NRA).intersect(theories).isNotEmpty()) {
                return false
            }
        }

        if (FP in theories) {
            if (setOf(LIA, NIA, NRA).intersect(theories).isNotEmpty()) {
                return false
            }
        }

        return true
    }
}

class KZ3SolverParamsConfiguration(params: Params) : KZ3SolverConfigurationImpl(params) {
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        throw KSolverException("Solver logic already configured")
    }
}

class KZ3SolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KZ3SolverConfiguration {
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        builder.buildOptimizeForTheories(theories, quantifiersAllowed)
    }

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
