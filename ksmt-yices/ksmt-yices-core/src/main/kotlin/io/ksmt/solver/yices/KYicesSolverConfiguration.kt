package io.ksmt.solver.yices

import com.sri.yices.Config
import io.ksmt.solver.KSolverConfiguration
import io.ksmt.solver.KSolverUniversalConfigurationBuilder
import io.ksmt.solver.KSolverUnsupportedFeatureException
import io.ksmt.solver.KSolverUnsupportedParameterException
import io.ksmt.solver.KTheory
import io.ksmt.solver.KTheory.FP
import io.ksmt.solver.KTheory.S
import io.ksmt.solver.KTheory.LIA
import io.ksmt.solver.KTheory.LRA
import io.ksmt.solver.KTheory.NIA
import io.ksmt.solver.KTheory.NRA
import io.ksmt.solver.smtLib2String

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
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        if (theories.isNullOrEmpty()) return

        if (FP in theories) {
            throw KSolverUnsupportedFeatureException("Unsupported theory $FP")
        }

        if (S in theories) {
            throw KSolverUnsupportedFeatureException("Unsupported theory $S")
        }

        // Yices requires MCSAT for the arithmetic theories
        if (setOf(LIA, LRA, NIA, NRA).intersect(theories).isNotEmpty()) {
            return
        }

        val theoryStr = theories.smtLib2String(quantifiersAllowed)
        config.defaultConfigForLogic(theoryStr)
    }

    override fun setYicesOption(option: String, value: String) {
        config.set(option, value)
    }
}

class KYicesSolverUniversalConfiguration(
    private val builder: KSolverUniversalConfigurationBuilder
) : KYicesSolverConfiguration {
    override fun optimizeForTheories(theories: Set<KTheory>?, quantifiersAllowed: Boolean) {
        builder.buildOptimizeForTheories(theories, quantifiersAllowed)
    }

    override fun setYicesOption(option: String, value: String) {
        builder.buildStringParameter(option, value)
    }
}
