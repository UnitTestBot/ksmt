package io.ksmt.solver.z3

import com.microsoft.z3.Params
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverException
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.Array
import io.ksmt.solver.KTheory.BV
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.KTheory.UF
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
        if (theories.isNullOrEmpty() || !supportedLogicCombination(theories, quantifiersAllowed)) {
            logicConfiguration = null
            return
        }

        logicConfiguration = theories.smtLib2String(quantifiersAllowed)
    }

    /**
     * Z3 provide special solver only for the following theory combinations
     * */
    private fun supportedLogicCombination(theories: Set<KTheory>, quantifiersAllowed: Boolean): Boolean =
        if (quantifiersAllowed) {
            theories in supportedTheoriesWithQuantifiers
        } else {
            theories in supportedQuantifierFreeTheories
        }

    companion object {
        private fun l(vararg theories: KTheory) = theories.toSet()

        private val supportedTheoriesWithQuantifiers = setOf(
            l(Array, BV),
            l(Array, LIA),
            l(Array, UF, BV),
            l(Array, UF, LIA),
            l(Array, UF, LIA, LRA),
            l(Array, UF, NIA),
            l(Array, UF, NIA, NRA),
            l(BV),
            l(FP),
            l(LIA),
            l(LRA),
            l(NIA),
            l(NRA),
            l(UF),
            l(UF, BV),
            l(UF, LIA),
            l(UF, LRA),
            l(UF, NIA),
            l(UF, NIA, NRA),
            l(UF, NRA),
        )

        private val supportedQuantifierFreeTheories = setOf(
            l(Array),
            l(Array, BV),
            l(Array, LIA),
            l(Array, NIA),
            l(Array, UF, BV),
            l(Array, UF, LIA),
            l(Array, UF, LIA, LRA),
            l(Array, UF, NIA),
            l(Array, UF, NIA, NRA),
            l(BV),
            l(BV, FP),
            l(FP),
            l(FP, LRA),
            l(LIA),
            l(LIA, LRA),
            l(LRA),
            l(NIA),
            l(NIA, NRA),
            l(NRA),
            l(UF),
            l(UF, BV),
            l(UF, LIA),
            l(UF, LRA),
            l(UF, NIA),
            l(UF, NIA, NRA),
            l(UF, NRA),
        )
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
